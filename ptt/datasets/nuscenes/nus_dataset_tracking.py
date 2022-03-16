# -*- coding: utf-8 -*-
import ipdb
import torch
import pickle
import numpy as np
import os
import pandas as pd
import nuscenes

from pyquaternion import Quaternion
from tqdm import tqdm
from collections import defaultdict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from ptt.datasets.tracking_dataset import DatasetTemplate
from ptt.datasets.kitti import kitti_tracking_utils as utils
from ptt.datasets.nuscenes import nus_splits


if torch.cuda.get_device_name(0) not in ['GeForce RTX 2080 Ti', 'GeForce RTX 3090']:
    from tools.visual_utils.visualize_utils import mayavi_show_np


class NuscenesTrackingDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.version = self.dataset_cfg.VERSION
        self.key_frame_only = self.dataset_cfg.KEY_FRAME_ONLY
        self.min_points = self.dataset_cfg.INIT_POINTS_THRESHOLD if self.mode == 'test' else -1
        self.preload_offset = self.dataset_cfg.LIDAR_CROP_OFFSET if self.mode == 'train' else -1
        self.sample_interval = self.dataset_cfg.SAMPLED_INTERVAL
        self.num_candidates_perframe = self.dataset_cfg.NUM_CANDIDATES_PERFRAME
        self.nusc = NuScenes(version=self.version, dataroot=self.root_path, verbose=False)

        self.track_instances = self.filter_instance(self.split, class_names.lower(), self.min_points)
        self.per_sequence_anno, self.seq_len_list = self._build_tracklet_anno()
        self.frame_seq_map = self.get_frame_seq_map()   # 0 - n-1

        self.database = []
        self.running_memory = {}
        self.lidar_frames = {}

        if self.load_from_db:
            self.load_from_database()

    def filter_instance(self, split, category_name=None, min_points=-1):
        """from the code of BAT"""
        if category_name is not None:
            general_classes = nus_splits.tracking_to_general_class[category_name]
        instances = []
        scene_splits = nus_splits.create_splits_scenes()
        for instance in self.nusc.instance:
            anno = self.nusc.get('sample_annotation', instance['first_annotation_token'])
            sample = self.nusc.get('sample', anno['sample_token'])
            scene = self.nusc.get('scene', sample['scene_token'])
            instance_category = self.nusc.get('category', instance['category_token'])['name']
            if scene['name'] in scene_splits[split] and anno['num_lidar_pts'] >= min_points and \
                    (category_name is None or category_name is not None and instance_category in general_classes):
                instances.append(instance)
        return instances

    def _build_tracklet_anno(self):
        list_of_tracklet_anno = []
        list_of_tracklet_len = []
        for instance in self.track_instances:
            track_anno = []
            curr_anno_token = instance['first_annotation_token']

            while curr_anno_token != '':
                ann_record = self.nusc.get('sample_annotation', curr_anno_token)
                sample = self.nusc.get('sample', ann_record['sample_token'])
                sample_data_lidar = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

                curr_anno_token = ann_record['next']
                if self.key_frame_only and not sample_data_lidar['is_key_frame']:
                    continue
                track_anno.append({"sample_data_lidar": sample_data_lidar, "box_anno": ann_record})

            list_of_tracklet_anno.append(track_anno)
            list_of_tracklet_len.append(len(track_anno))
        return list_of_tracklet_anno, list_of_tracklet_len

    def grab_data(self, tracklet_id, frame_id):
        if self.load_from_db:
            data = self.database[tracklet_id][frame_id]
        else:
            cur_frame_anno = self.per_sequence_anno[tracklet_id][frame_id]
            data = self.get_data_from_anno(cur_frame_anno)

        data['anno'].update({"scene_num": -1, "frame_num": -1, "track_id": -1})
        return data

    def load_from_database(self):
        database_path = self.dataset_cfg.INFO_PATH[self.mode]
        prefix = [self.class_names, self.dataset_cfg.REF_COOR, str(self.preload_offset)]
        database_path = '_'.join(
            [database_path.split('_')[0]] + prefix + database_path.split('_')[1:])
        database_path = self.root_path / database_path
        self.logger('database is at %s' % database_path)
        if not database_path.exists():
            self.logger('generating database... start')
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
                self.database = pickle.load(f)

    def get_data_from_anno(self, anno):
        box = self.get_box(anno)
        pc = self.get_lidar(anno, box)
        return {'pc': pc, 'box': box, 'anno': anno}

    def get_box(self, anno):
        box_anno = anno['box_anno']
        bb = Box(box_anno['translation'], box_anno['size'], Quaternion(box_anno['rotation']),
                 name=box_anno['category_name'], token=box_anno['token'])
        return bb

    def get_lidar(self, anno, box):
        sample_data_lidar = anno['sample_data_lidar']
        # {'token': '80a35c14dd68408d83cf0e4f814feae4', 'sample_token': 'fd8420396768425eabec9bdddf7e64b6',
        #  'ego_pose_token': '80a35c14dd68408d83cf0e4f814feae4',
        #  'calibrated_sensor_token': '53a38cc5fb2a491b83d9b18a5071e12a', 'timestamp': 1533201470448696,
        #  'fileformat': 'pcd', 'is_key_frame': True, 'height': 0, 'width': 0,
        #  'filename': 'samples/LIDAR_TOP/n015-2018-08-02-17-16-37+0800__LIDAR_TOP__1533201470448696.pcd.bin',
        #  'prev': '', 'next': 'e168f3fa0f9e4956aecca639b0f52555', 'sensor_modality': 'lidar', 'channel': 'LIDAR_TOP'}
        lidar_file = os.path.join(self.root_path, sample_data_lidar['filename'])
        pc = LidarPointCloud.from_file(lidar_file)
        cs_record = self.nusc.get('calibrated_sensor', sample_data_lidar['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        pose_record = self.nusc.get('ego_pose', sample_data_lidar['ego_pose_token'])
        pc.rotate(Quaternion(pose_record['rotation']).rotation_matrix)
        pc.translate(np.array(pose_record['translation']))

        pc = utils.PointCloud(points=pc.points)
        if self.preload_offset > 0:
            pc = utils.crop_pc(pc, box, offset=self.preload_offset)

        return pc

    def __len__(self):
        total = sum(self.seq_len_list) * self.num_candidates_perframe // self.sample_interval \
            if self.training else len(self.per_sequence_anno)
        return total

    def __getitem__(self, index):
        if self.mode == 'train':
            index *= self.sample_interval
            ret_dict = self.get_train_items(index)  # 'search_points', 'template_points', 'cls_label', 'reg_label'])
            # ret_dict = self.point_feature_encoder.forward(ret_dict)
            # if self.data_augmentor:
            #     ret_dict = self.data_augmentor.forward(data_dict=ret_dict)   # only for lidar coord
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

        # assert first_frame_data['anno']['track_id'] == cur_frame_data['anno']['track_id']
        # assert pre_frame_data['anno']['track_id'] == cur_frame_data['anno']['track_id']

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
        return sum(self.seq_len_list)

    @property
    def num_tracklets(self):
        return len(self.per_sequence_anno)


if __name__ == '__main__':
    import sys
    import yaml
    from pathlib import Path
    from easydict import EasyDict

    dataset_cfg = EasyDict(yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)).DATA_CONFIG
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    dataset = NuscenesTrackingDataset(dataset_cfg=dataset_cfg, class_names='Car',
                                   root_path=ROOT_DIR / 'data' / 'nuScenes',
                                   training=True)
    for i in range(200):
        dataset[i]
    # import ipdb; ipdb.set_trace()
    # dataset = SiameseTrain(input_size=1024, path=ROOT_DIR / 'data' / 'kitti_tracking/' / 'training/', split='traintiny')
    # for i in range(200):
    #     dataset[i]
