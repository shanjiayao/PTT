import time
import numpy as np
import torch
import tqdm
import copy

from pyquaternion import Quaternion
from ptt.utils import timer_utils as timer
from ptt.utils.common_utils import MovingAverage
from ptt.datasets.kitti.kitti_tracking_utils import PointCloud, get_box_by_offset
from ptt.datasets.kitti import kitti_tracking_utils as kitti_utils
from ptt.utils.file_io import save_track_results
from tools.eval_utils.eval_tracking_metrics import Evaluator, AverageMeter


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False,
                   result_dir=None, tb_log=None):
    result_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = result_dir / 'final_result' / 'data'
    final_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    model.eval()

    progress_bar = tqdm.tqdm(total=dataloader.dataset.num_frames, leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    evaluator = TrackingEvaluator(cfg_=cfg, logger_=logger, model_=model, timer_=timer,
                                  dataset_=dataloader.dataset, output_dir=final_output_dir)

    evaluator.ret_dict.update({'batch_num': 0})
    for batch_ in dataloader:
        with torch.no_grad():
            evaluator.ret_dict['batch_num'] += 1
            evaluator.test_batch(batch_, progress_bar)
    succ, prec = evaluator.log_succ_prec()
    progress_bar.close()

    if tb_log:
        try:
            tb_log.add_scalars('metric', {'succ': succ, 'prec': prec}, epoch_id)
        except ValueError:
            pass

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)

    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    logger.info('****************Evaluation done.*****************')
    return succ, prec


class TrackingEvaluator:
    def __init__(self, cfg_, logger_, timer_, model_, dataset_, output_dir):
        self.logger = logger_ if logger_ else print
        self.timer = timer_
        self.cfg = cfg_
        self.model = model_
        self.dataset = dataset_

        self.Success_run = AverageMeter()
        self.Precision_run = AverageMeter()
        self.evaluator = Evaluator(cfg_=cfg_)
        self.ret_dict = {}
        self.result_file = output_dir / "track_result.txt"
        self.fp = open(self.result_file, 'w')

    def log_succ_prec(self):
        self.Success_run.update(self.evaluator.Success_main.average)
        self.Precision_run.update(self.evaluator.Precision_main.average)
        self.logger.info(
            "mean Succ/Prec {:.1f}/{:.1f}".format(self.Success_run.avg, self.Precision_run.avg))
        self.fp.close()
        return self.Success_run.avg, self.Precision_run.avg

    def test_batch(self, batch, tbar):
        for PCs, BBs, list_of_anno in batch:  # tracklet
            self.ret_dict.update({"results_BBs": []})
            with self.evaluator:
                for i, _ in enumerate(PCs):
                    this_anno = list_of_anno[i]
                    try:
                        self.ret_dict.update({
                            "scene_num": this_anno[0], "frame_num": this_anno[1],
                            "track_id": this_anno[2], "this_BB": BBs[i], "this_PC": PCs[i],
                            "PCs": PCs, "BBs": BBs
                        })
                    except:  # for nuscenes
                        self.ret_dict.update({
                            "scene_num": -1, "frame_num": -1,
                            "track_id": -1, "this_BB": BBs[i], "this_PC": PCs[i],
                            "PCs": PCs, "BBs": BBs
                        })

                    if i == 0:   # first frame
                        with self.timer.env('everything else'):
                            box = self.ret_dict["this_BB"]
                            self.ret_dict["results_BBs"].append(box)
                        self.tracker_initialize()
                    else:
                        avg = MovingAverage()
                        self.timer.reset()
                        with self.timer.env('everything else'):
                            self.test_frame(i)
                            avg.add(self.timer.total_time())
                            self.timer.print_stats()
                            print('Avg fps: %.2f     Avg ms: %.2f         ' % (1 / avg.get_avg(), avg.get_avg() * 1000))
                            if self.cfg.TEST.VISUALIZE:
                                self.mayavi_show()

                    self.evaluator.update_iou(self.ret_dict['this_BB'], self.ret_dict['results_BBs'][-1])
                    tbar.update(1)
                    tbar.set_description('batch {}  '.format(self.ret_dict["batch_num"]) +
                                         'batch Succ/Prec:' +
                                         '{:.1f}/'.format(self.evaluator.Success_batch.average) +
                                         '{:.1f}'.format(self.evaluator.Precision_batch.average))
                    self.save_track_resluts()
                    self.save_pts_pcd()
                self.batch_log()

    def tracker_initialize(self):
        candidate_pc, candidate_label, _ = kitti_utils.crop_center_pc(
            self.ret_dict["this_PC"], self.ret_dict["this_BB"], self.ret_dict["this_BB"],
            offset=self.dataset.dataset_cfg.SEARCH_BB_OFFSET,
            scale=self.dataset.dataset_cfg.SEARCH_BB_SCALE
        )
        candidate_pcs = kitti_utils.regularize_pc(
            candidate_pc, self.dataset.dataset_cfg.SEARCH_INPUT_SIZE, istrain=False
        )
        candidate_pcs_torch = torch.from_numpy(candidate_pcs).float().cuda()
        candidate_pcs_torch = candidate_pcs_torch.unsqueeze(0).cuda()

        self.ret_dict.update({
            "candidate_PC": candidate_pcs_torch,
            "model_points": candidate_pc.points.T,
            "model_label": candidate_label
        })

    def test_frame(self, frame):
        torch.cuda.synchronize()
        with self.timer.env("pre process"):
            self.prepare_search(frame)
            self.prepare_template(frame)

        torch.cuda.synchronize()
        with self.timer.env("model inference"):
            self.model_inference()

        torch.cuda.synchronize()
        with self.timer.env("post process"):
            self.post_process()

    def prepare_search(self, frame_id, debug=False):
        if "previous_result".upper() in self.cfg.TEST.REF_BOX.upper():
            ref_box = self.ret_dict["results_BBs"][-1]
        elif "previous_gt".upper() in self.cfg.TEST.REF_BOX.upper():
            ref_box = self.ret_dict["BBs"][frame_id - 1]
        elif "current_gt".upper() in self.cfg.TEST.REF_BOX.upper():
            ref_box = self.ret_dict["this_BB"]
        else:
            raise ValueError("reference_BB must be set with previous_result/previous_gt/current_gt")

        candidate_pc, candidate_label, _ = kitti_utils.crop_center_pc(
            pc=self.ret_dict["this_PC"], sample_box=ref_box,
            gt_box=self.ret_dict["this_BB"],
            offset=self.dataset.dataset_cfg.SEARCH_BB_OFFSET,
            scale=self.dataset.dataset_cfg.SEARCH_BB_SCALE
        )
        candidate_pcs = kitti_utils.regularize_pc(
            pc=candidate_pc, input_size=self.dataset.dataset_cfg.SEARCH_INPUT_SIZE, istrain=False
        )
        candidate_pcs_torch = torch.from_numpy(candidate_pcs).float().cuda()
        candidate_pcs_torch = candidate_pcs_torch.unsqueeze(0).cuda()

        if debug:
            from tools.visual_utils.visualize_utils import mayavi_show_np
            mayavi_show_np(copy.deepcopy(candidate_pc.points.T))

        self.ret_dict.update({
            "ref_BB": ref_box,
            "candidate_PC": candidate_pcs_torch,
            "candidate_points": candidate_pc.points.T,
            "candidate_label": candidate_label
        })

    def prepare_template(self, frame_id, debug=False):
        if "firstandprevious".upper() in self.cfg.TEST.SHAPE_AGGREGATION.upper():
            model_pc = kitti_utils.get_model(
                [self.ret_dict["PCs"][0], self.ret_dict["PCs"][frame_id - 1]],
                [self.ret_dict["results_BBs"][0], self.ret_dict["results_BBs"][frame_id - 1]],
                offset=self.dataset.dataset_cfg.MODEL_BB_OFFSET,
                scale=self.dataset.dataset_cfg.MODEL_BB_SCALE
            )
        elif "first".upper() in self.cfg.TEST.SHAPE_AGGREGATION.upper():
            model_pc = kitti_utils.get_model(
                [self.ret_dict["PCs"][0]],
                [self.ret_dict["results_BBs"][0]],
                offset=self.dataset.dataset_cfg.MODEL_BB_OFFSET,
                scale=self.dataset.dataset_cfg.MODEL_BB_SCALE
            )
        elif "previous".upper() in self.cfg.TEST.SHAPE_AGGREGATION.upper():
            model_pc = kitti_utils.get_model(
                [self.ret_dict["PCs"][frame_id - 1]],
                [self.ret_dict["results_BBs"][frame_id - 1]],
                offset=self.dataset.dataset_cfg.MODEL_BB_OFFSET,
                scale=self.dataset.dataset_cfg.MODEL_BB_SCALE
            )
        elif "all".upper() in self.cfg.TEST.SHAPE_AGGREGATION.upper():
            model_pc = kitti_utils.get_model(
                self.ret_dict["PCs"][:frame_id],
                self.ret_dict["results_BBs"],
                offset=self.dataset.dataset_cfg.MODEL_BB_OFFSET,
                scale=self.dataset.dataset_cfg.MODEL_BB_SCALE
            )
        else:
            model_pc = kitti_utils.get_model(
                self.ret_dict["PCs"][:frame_id],
                self.ret_dict["results_BBs"],
                offset=self.dataset.dataset_cfg.MODEL_BB_OFFSET,
                scale=self.dataset.dataset_cfg.MODEL_BB_SCALE
            )

        model_pc = kitti_utils.regularize_pc(model_pc, self.dataset.dataset_cfg.TEMPLATE_INPUT_SIZE, istrain=False)
        model_pc_torch = torch.from_numpy(model_pc).float().cuda().unsqueeze(0)
        if debug:
            from tools.visual_utils.visualize_utils import mayavi_show_np
            mayavi_show_np(copy.deepcopy(model_pc))
        self.ret_dict.update({"model_PC": model_pc_torch})

    def model_inference(self):
        input_dict = {
            'search_points': self.ret_dict["candidate_PC"],  # 1024, 4
            'template_points': self.ret_dict["model_PC"],  # 512, 4
        }

        end_points = self.model(input_dict)
        if self.cfg.MODEL.NAME != 'Pointnet_Tracking':
            seed_feat = end_points['search_feats']
            seed_xyz = end_points['search_seeds']
            vote_score = end_points['pred_centroids_cls']
            vote_xyz = end_points['pred_centroids_votes']
            estimation_box = end_points['pred_box_data']
            center_xyz = end_points['pred_box_center']
        else:  # fix the error from old model
            seed_feat = end_points['search_feat']
            seed_xyz = end_points['search_xyz']
            vote_score = end_points['pred_cla_test']
            estimation_cla = end_points['pred_cla']
            vote_xyz = end_points['pred_reg']
            estimation_box = end_points['pred_box_data']
            center_xyz = end_points['pred_box_center']

        model_output = {
            'seed_feat': seed_feat,
            'seed_xyz': seed_xyz,
            'pred_cls': vote_score,
            'pred_vote': vote_xyz,
            'pred_box': estimation_box,
            'pred_xyz': center_xyz
        }
        self.ret_dict.update({
            "model_output": model_output
        })

    def post_process(self):
        estimation_boxs_cpu = self.ret_dict['model_output']['pred_box'].squeeze(0).detach().cpu().numpy()
        box_idx = estimation_boxs_cpu[:, 4].argmax()
        estimation_box_cpu = estimation_boxs_cpu[box_idx, 0:4]
        box = get_box_by_offset(self.ret_dict['ref_BB'], estimation_box_cpu, self.cfg.DATA_CONFIG.USE_Z_AXIS)
        self.ret_dict.update({
            'proposal_score': estimation_boxs_cpu[box_idx, 4]
        })
        self.ret_dict["results_BBs"].append(box)

    def save_track_resluts(self):
        box = copy.deepcopy(self.ret_dict["results_BBs"][-1])
        save_track_results(self.fp,
                           [self.ret_dict["scene_num"], self.ret_dict["frame_num"], self.ret_dict["batch_num"]],
                           box.corners().transpose())

    def save_pts_pcd(self):
        if self.cfg.TEST.SAVE_PCD:
            from ptt.utils.file_io import save_pts_as_pcd
            points = copy.deepcopy(self.ret_dict["candidate_PC"].squeeze(0).detach().cpu().numpy())
            pc = PointCloud(points.T)
            if len(self.ret_dict["results_BBs"]) > 1:
                ref_box = copy.deepcopy(self.ret_dict["results_BBs"][-2])
            else:
                ref_box = copy.deepcopy(self.ret_dict["results_BBs"][-1])
            trans = ref_box.center
            rot_mat = ref_box.rotation_matrix
            pc.rotate(rot_mat)
            pc.translate(trans)
            save_pts_as_pcd(points=pc.points.T,
                            path="../output/pcd",
                            name=str(self.ret_dict["scene_num"]) + "_" +
                                 str(self.ret_dict["track_id"]) + "_candidatePC_" + str(
                                self.ret_dict["frame_num"]) + ".pcd")

    def batch_log(self):
        self.logger.info('batch {}  '.format(self.ret_dict["batch_num"]) +
                         'batch Succ/Prec:' +
                         '|{:.1f}|/'.format(self.evaluator.Success_batch.average) +
                         '{:.1f} '.format(self.evaluator.Precision_batch.average) +
                         'all_pts|{}| '.format(self.ret_dict['model_points'].shape[0]) +
                         'fore_pts|{}|'.format(np.sum(self.ret_dict['model_label'] == 1)))

    def mayavi_show(self):
        pass


if __name__ == '__main__':
    pass
