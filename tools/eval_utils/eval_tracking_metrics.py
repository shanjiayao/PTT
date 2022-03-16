import numpy as np
from shapely.geometry import Polygon
from ptt.utils.common_utils import AverageMeter


class Evaluator:
    def __init__(self, cfg_=None, timer_=None):
        self.timer = timer_
        self.cfg = cfg_
        self.ref_coordinate = cfg_.DATA_CONFIG.REF_COOR
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()

        self.Success_main = Success()
        self.Precision_main = Precision()
        self.Success_batch = Success()
        self.Precision_batch = Precision()

    def update_iou(self, gt, pred, iou_dims=3):
        this_overlap = estimateOverlap(gt, pred, dim=iou_dims, ref_coord=self.ref_coordinate)
        print("-> 3D IOU is {: 2.2f}%".format(this_overlap * 100))

        this_accuracy = estimateAccuracy(gt, pred, dim=iou_dims)
        self.Success_main.add_overlap(this_overlap)
        self.Precision_main.add_accuracy(this_accuracy)
        self.Success_batch.add_overlap(this_overlap)
        self.Precision_batch.add_accuracy(this_accuracy)

    def __enter__(self):
        pass

    def __exit__(self, e, ev, t):
        self.Success_batch.reset()
        self.Precision_batch.reset()


def estimateAccuracy(box_a, box_b, dim=3):
    if dim == 3:
        return np.linalg.norm(box_a.center - box_b.center, ord=2)
    elif dim == 2:
        return np.linalg.norm(
            box_a.center[[0, 2]] - box_b.center[[0, 2]], ord=2)


def fromBoxToPoly(box, ref_coord):
    if ref_coord.lower() == 'camera':
        return Polygon(tuple(box.corners()[[0, 2]].T[[0, 1, 5, 4]]))
    elif ref_coord.lower() == 'lidar':
        return Polygon(tuple(box.bottom_corners().T))


def estimateOverlap(box_a, box_b, dim=2, ref_coord='camera'):
    if box_a == box_b:
        return 1.0

    poly_anno = fromBoxToPoly(box_a, ref_coord)
    poly_subm = fromBoxToPoly(box_b, ref_coord)

    box_inter = poly_anno.intersection(poly_subm)
    box_union = poly_anno.union(poly_subm)

    if dim == 2:
        return box_inter.area / box_union.area
    else:
        ymax = min(box_a.center[1], box_b.center[1])
        ymin = max(box_a.center[1] - box_a.wlh[2],
                   box_b.center[1] - box_b.wlh[2])

        inter_vol = box_inter.area * max(0, ymax - ymin)
        anno_vol = box_a.wlh[0] * box_a.wlh[1] * box_a.wlh[2]
        subm_vol = box_b.wlh[0] * box_b.wlh[1] * box_b.wlh[2]

        overlap = inter_vol * 1.0 / (anno_vol + subm_vol - inter_vol)
    return overlap


class Success(object):
    def __init__(self, n=21, max_overlap=1):
        self.max_overlap = max_overlap
        self.Xaxis = np.linspace(0, self.max_overlap, n)
        self.reset()

    def reset(self):
        self.overlaps = []

    def add_overlap(self, val, index=None):
        self.overlaps.append(val)

    @property
    def get_main(self):
        main_avg = [
            np.sum(i for i in self.overlaps).astype(float) / self.count
        ]
        return main_avg

    @property
    def count(self):
        return len(self.overlaps)

    @property
    def value(self):
        succ = [
            np.sum(i >= thres
                   for i in self.overlaps).astype(float) / self.count
            for thres in self.Xaxis
        ]
        return np.array(succ)

    @property
    def average(self):
        if len(self.overlaps) == 0:
            return 0
        return np.trapz(self.value, x=self.Xaxis) * 100 / self.max_overlap


class Precision(object):
    """Computes and stores the Precision"""

    def __init__(self, n=21, max_accuracy=2):
        self.max_accuracy = max_accuracy
        self.Xaxis = np.linspace(0, self.max_accuracy, n)
        self.reset()

    def reset(self):
        self.accuracies = []

    def add_accuracy(self, val, index=None):
        self.accuracies.append(val)

    @property
    def get_main(self):
        main_avg = [
            np.sum(i for i in self.accuracies).astype(float) / self.count
        ]
        return main_avg

    @property
    def count(self):
        return len(self.accuracies)

    @property
    def value(self):
        prec = [
            np.sum(i <= thres
                   for i in self.accuracies).astype(float) / self.count
            for thres in self.Xaxis
        ]
        return np.array(prec)

    @property
    def average(self):
        if len(self.accuracies) == 0:
            return 0
        return np.trapz(self.value, x=self.Xaxis) * 100 / self.max_accuracy
