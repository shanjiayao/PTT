import torch
import pickle
import numpy as np
import os
import pandas as pd

from pyquaternion import Quaternion
from skimage import io
from tqdm import tqdm
from collections import defaultdict
from ptt.datasets.tracking_dataset import DatasetTemplate
from ptt.datasets.kitti import kitti_tracking_utils as utils
from ptt.utils import calibration_kitti_tracking, track3d_kitti

if torch.cuda.get_device_name(0) not in ['GeForce RTX 2080 Ti', 'GeForce RTX 3090']:
    from tools.visual_utils.visualize_utils import mayavi_show_np


class KittiTrackingDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / 'training'
        self.sample_interval = self.dataset_cfg.SAMPLED_INTERVAL
        self.num_candidates_perframe = self.dataset_cfg.NUM_CANDIDATES_PERFRAME

        self.tracklet_list = []
        self.anno_list = []
        self.running_memory = {}
        self.lidar_frames = defaultdict(dict)

        self.per_sequence_anno = self.get_tracklet_list(self.get_scenes(self.split))  # sequence-level info

        self.per_frame_anno = [anno for tracklet_anno in self.per_sequence_anno
                               for anno in tracklet_anno]  # frame-level
        self.frame_seq_map = self.get_frame_seq_map()   # 0 - n-1
        self.database = []
        self.preload_offset = self.dataset_cfg.LIDAR_CROP_OFFSET if self.mode == 'train' else -1
        if self.load_from_db:
            self.load_from_database()

    def __len__(self):
        total = len(self.per_frame_anno) * self.num_candidates_perframe // self.sample_interval \
            if self.training else len(self.per_sequence_anno)
        return total

    def __getitem__(self, index):
        if self.mode == 'train':
            index *= self.sample_interval
            ret_dict = self.get_train_items(index)  # 'search_points', 'template_points', 'cls_label', 'reg_label'])
            ret_dict = self.point_feature_encoder.forward(ret_dict)
            if self.data_augmentor:
                ret_dict = self.data_augmentor.forward(data_dict=ret_dict)   # only for lidar coord
            return ret_dict
        else:
            return self.get_test_items(index)

    def get_train_items(self, index):
        data_dict = {}
        """prepare current points and label"""
        anno_index = self.get_anno_index(index)
        aug_index = self.get_aug_index(index)

        tracklet_id, frame_id = self.frame_seq_map[anno_index]
        cur_frame_data = self.grab_data(tracklet_id, frame_id)

        if self.debug:
            print('this frame and gt box')
            mayavi_show_np(cur_frame_data['pc'].points.T, box=cur_frame_data['box'])

        tensor_pc, tensor_cls_gt, tensor_reg_gt = self.prepare_search_and_label(cur_frame_data, aug_index)

        if isinstance(tensor_pc, bool):
            return self.get_train_items(np.random.randint(0, self.__len__()))

        data_dict.update({
            'search_pts': tensor_pc,
            'cls_label': tensor_cls_gt,
            'reg_label': tensor_reg_gt
        })

        """prepare template points and label"""
        first_frame_data = self.grab_data(tracklet_id, 0)
        pre_frame_data = self.grab_data(tracklet_id, max(frame_id - 1, 0))
        assert first_frame_data['anno']['track_id'] == cur_frame_data['anno']['track_id']
        assert pre_frame_data['anno']['track_id'] == cur_frame_data['anno']['track_id']

        if self.debug:
            print('previous frame and gt box')
            mayavi_show_np(pre_frame_data['pc'].points.T, box=pre_frame_data['box'])
            print('first frame and gt box')
            mayavi_show_np(first_frame_data['pc'].points.T, box=first_frame_data['box'])

        data_dict['template_pts'] = self.prepare_template_data([first_frame_data, pre_frame_data], aug_index)

        if isinstance(data_dict['template_pts'], bool):
            return self.get_train_items(np.random.randint(0, self.__len__()))

        ret_dict = {
            'search_points': data_dict['search_pts'],
            'template_points': data_dict['template_pts'],
            'cls_label': data_dict['cls_label'],
            'reg_label': data_dict['reg_label'],
        }
        return ret_dict

    def get_test_items(self, index):
        cur_seq_anno = self.per_sequence_anno[index]
        frame_ids = list(range(len(cur_seq_anno)))
        pcs = []
        bboxes = []
        for idx in frame_ids:
            frame_dict = self.grab_data(index, idx)
            pcs.append(frame_dict['pc'])
            bboxes.append(frame_dict['box'])
        return pcs, bboxes, cur_seq_anno

    def prepare_search_and_label(self, data_dict, offset_id):
        if offset_id == 0:
            sample_offsets = np.zeros(3)
        else:
            gaussian = utils.KalmanFiltering(bnd=[1, 1, 5])
            sample_offsets = gaussian.sample(1)[0]

        # sample_offsets : x, y, z  and z also be used as theta offset
        sample_box = utils.get_box_by_offset(data_dict['box'], sample_offsets, self.dataset_cfg.USE_Z_AXIS)
        sample_pc, sample_label, sample_reg = utils.crop_center_pc(
            pc=data_dict['pc'],
            sample_box=sample_box,
            gt_box=data_dict['box'],
            sample_offsets=sample_offsets,
            offset=self.dataset_cfg.SEARCH_BB_OFFSET,
            scale=self.dataset_cfg.SEARCH_BB_SCALE,
            visual_handle=mayavi_show_np if self.debug else None,
            refine_box=self.dataset_cfg.REFINE_BOX_SIZE
        )

        if sample_pc.nbr_points() <= 20:
            return False, False, False

        sample_pts, cls_gt, reg_gt = utils.regularize_pc(
            pc=sample_pc,
            label=sample_label,
            reg=sample_reg,
            input_size=self.dataset_cfg.SEARCH_INPUT_SIZE
        )
        return sample_pts, cls_gt, reg_gt

    def prepare_template_data(self, frames_data_list, offset_id, ):
        if offset_id == 0:
            sample_offsets = np.zeros(3)
        else:
            sample_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
            sample_offsets[2] = sample_offsets[2] * 5.0

        pcs = [frame_data['pc'] for frame_data in frames_data_list]
        boxes = [frame_data['box'] for frame_data in frames_data_list]
        boxes[-1] = utils.get_box_by_offset(boxes[-1], sample_offsets, self.dataset_cfg.USE_Z_AXIS)

        if self.debug:
            print('[template] two pcs and boxes')
            print(sample_offsets)
            mayavi_show_np(pcs[0].points.T, box=boxes[0])
            mayavi_show_np(pcs[1].points.T, box=boxes[1])

        template_pc = utils.get_model(
            PCs=pcs,
            boxes=boxes,
            offset=self.dataset_cfg.MODEL_BB_OFFSET,
            scale=self.dataset_cfg.MODEL_BB_SCALE,
            visual_handle=mayavi_show_np if self.debug else None
        )

        if template_pc.nbr_points() <= 20:
            return False
        template_pts = utils.regularize_pc(template_pc, self.dataset_cfg.TEMPLATE_INPUT_SIZE)
        return template_pts

    def grab_data(self, tracklet_id, frame_id):
        if self.load_from_db:
            return self.database[tracklet_id][frame_id]
        else:
            cur_frame_anno = self.per_sequence_anno[tracklet_id][frame_id]
            return self.get_data_from_anno(cur_frame_anno)

    def load_from_database(self):
        database_path = self.dataset_cfg.INFO_PATH[self.mode]
        prefix = [self.class_names, self.dataset_cfg.REF_COOR, str(self.preload_offset)]
        database_path = '_'.join([database_path.split('_')[0]] + prefix + database_path.split('_')[1:])
        database_path = self.root_path / database_path
        if not database_path.exists():
            self.logger('generating database... start')
            self.logger('database generate at %s' % database_path)

            for k in tqdm(range(len(self.per_sequence_anno))):
                frame_data = []
                for anno in self.per_sequence_anno[k]:
                    frame_data.append(self.get_data_from_anno(anno))
                self.database.append(frame_data)
            with open(database_path, 'wb') as f:
                pickle.dump(self.database, f)
            self.logger('database generated at %s' % database_path)
        else:
            self.logger('load from database at %s' % database_path)
            with open(database_path, 'rb') as f:
                # import ipdb; ipdb.set_trace()
                self.database = pickle.load(f)

    def get_frame_seq_map(self):
        id_map = {}
        cnt = 0
        for k in range(len(self.per_sequence_anno)):
            cur_seq_anno = self.per_sequence_anno[k]
            for n in range(len(cur_seq_anno)):
                id_map[cnt] = (k, n)
                cnt += 1
        return id_map

    def get_anno_index(self, index):
        return int(index / self.num_candidates_perframe)

    def get_aug_index(self, index):
        return int(index % self.num_candidates_perframe)

    @property
    def num_frames(self):
        return len(self.per_frame_anno)

    @property
    def num_tracklets(self):
        return len(self.per_sequence_anno)

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / 'training'

        scenes_list = self.get_scenes(self.split)
        self.tracklet_list = self.get_tracklet_list(scenes_list)  # sequence-level info
        if self.mode == 'train':
            for i in tqdm(range(len(self.tracklet_list))):
                tracklet_anno = self.tracklet_list[i]
                for j in range(len(tracklet_anno)):
                    tracklet_anno[j]['model_idx'] = i
                    tracklet_anno[j]['relative_idx'] = j

        self.anno_list = [anno for tracklet_anno in self.tracklet_list for anno in tracklet_anno]  # frame-level info

    @staticmethod
    def get_scenes(split):
        if "TRAIN" in split.upper():
            scenes = [0] if "TINY" in split.upper() else list(range(0, 17))
        elif "VAL" in split.upper():
            scenes = [3] if "TINY" in split.upper() else list(range(17, 19))
        elif "TEST" in split.upper():
            scenes = [0] if "TINY" in split.upper() else list(range(19, 21))
        else:
            scenes = list(range(21))
        return scenes

    def get_tracklet_list(self, sceneID):
        lidar_path = self.root_split_path / 'velodyne'
        label_path = self.root_split_path / "label_02"

        list_of_scene = [
            path for path in os.listdir(lidar_path)
            if os.path.isdir(os.path.join(lidar_path, path)) and int(path) in sceneID
        ]

        list_of_tracklet_anno = []
        for scene in list_of_scene:
            label_file = os.path.join(label_path, scene + ".txt")
            df = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occluded",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "rotation_y"
                ])
            df = df[df["type"] == self.class_names]
            df.insert(loc=0, column="scene", value=scene)
            for track_id in df.track_id.unique():
                df_tracklet = df[df["track_id"] == track_id]
                sorted_df_tracklet = df_tracklet.sort_values(by=['frame'])
                assert sorted_df_tracklet['frame'].to_list() == df_tracklet['frame'].to_list()
                df_tracklet = df_tracklet.reset_index(drop=True)
                tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                list_of_tracklet_anno.append(tracklet_anno)
        return list_of_tracklet_anno

    def get_lidar(self, anno, box):
        try:
            try:
                pc = self.lidar_frames[anno['scene']][anno['frame']]   # use this dict to reduce memory
            except KeyError:
                lidar_file = self.root_split_path / 'velodyne' / anno['scene'] / ('{:06}.bin'.format(anno['frame']))
                pc = utils.PointCloud(np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4).T)
                if self.ref_coor == 'CAMERA':
                    transf_mat = np.vstack((anno['V2C'], np.array([0, 0, 0, 1])))
                    pc.transform(transf_mat)
                self.lidar_frames[anno['scene']][anno['frame']] = pc
            if self.preload_offset > 0:
                pc = utils.crop_pc(pc, box, offset=self.preload_offset)
            return pc
        except:
            return utils.PointCloud(np.array([[0, 0, 0]]).T)

    def get_box(self, anno):
        wlh = [anno['width'], anno['length'], anno['height']]
        if self.ref_coor == 'LIDAR':
            orientation = Quaternion(axis=[0, 0, 1], radians=anno['rotation_y_lidar'])
            box = utils.Box(anno['ctr_in_lidar'], wlh, orientation)
        elif self.ref_coor == 'CAMERA':
            orientation = Quaternion(axis=[0, 1, 0], radians=anno['rotation_y']) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
            box = utils.Box(anno['ctr_in_camera'], wlh, orientation)
        else:
            box = None
            raise ValueError("ref_coor must be CAMERA or LIDAR!")
        return box

    def get_image_shape(self, anno):
        img_file = self.root_split_path / 'image_02' / anno['scene'] / ('%s.png' % anno['frame'])
        assert img_file.exists(), '\033[1;31m' + '\nimage file does not exists... \nExiting...' + '\033[0m'
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, anno):
        label_file = self.root_split_path / 'label_02' / (anno['scene'] + '.txt')
        assert label_file.exists(), '\033[1;31m' + str(label_file) + '\nlabel file does not exists... ' + '\033[0m'
        return track3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, anno):
        calib_file = self.root_split_path / 'calib' / (anno['scene'] + '.txt')
        assert calib_file.exists(), '\033[1;31m' + str(calib_file) + '\ncalib file does not exists...' + '\033[0m'
        return calibration_kitti_tracking.Calibration(calib_file)

    def get_data_from_anno(self, anno):
        def update_anno(anno_):
            calib_ = self.get_calib(anno_)
            box_center = np.array([anno_['x'], anno_['y'], anno_['z']]).reshape(1, 3)
            box_center_lidar_coor = calib_.project_rect_to_velo(box_center)
            box_center_lidar_coor[0, 2] += anno_['height'] / 2
            anno_['V2C'] = calib_.V2C
            anno_['ctr_in_camera'] = np.array([anno_['x'], anno_['y'] - anno_["height"] / 2, anno_['z']]).tolist()
            anno_['rotation_y_camera'] = anno_['rotation_y']
            anno_['ctr_in_lidar'] = box_center_lidar_coor[0].tolist()
            anno_['rotation_y_lidar'] = -(np.pi / 2 + anno_['rotation_y'])
            return anno_, calib_

        anno, calib = update_anno(anno)
        box = self.get_box(anno)
        pc = self.get_lidar(anno, box)
        return {'pc': pc, 'box': box, 'calib': calib, 'anno': anno}


if __name__ == '__main__':
    import sys
    import yaml
    from pathlib import Path
    from easydict import EasyDict

    dataset_cfg = EasyDict(yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)).DATA_CONFIG
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    dataset = KittiTrackingDataset(dataset_cfg=dataset_cfg, class_names='Car',
                                   root_path=ROOT_DIR / 'data' / 'kitti',
                                   training=False)
    for i in range(100):
        dataset[i]
