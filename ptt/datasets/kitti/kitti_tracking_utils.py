# -*- coding: utf-8 -*-
import torch
import numpy as np
import copy
from pyquaternion import Quaternion
from ptt.utils import common_utils


class PointCloud:
    def __init__(self, points):
        self.points = points
        if self.points.shape[0] > 3:
            self.points = self.points[0:3, :]

    @staticmethod
    def load_pcd_bin(file_name):
        scan = np.fromfile(file_name, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, :4]
        return points.T

    @classmethod
    def from_file(cls, file_name):
        if file_name.endswith('.bin'):
            points = cls.load_pcd_bin(file_name)
        elif file_name.endswith('.npy'):
            points = np.load(file_name)
        else:
            raise ValueError('Unsupported filetype {}'.format(file_name))

        return cls(points)

    def nbr_points(self):
        return self.points.shape[1]

    def subsample(self, ratio):
        selected_ind = np.random.choice(np.arange(0, self.nbr_points()), size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, radius):
        x_filt = np.abs(self.points[0, :]) < radius
        y_filt = np.abs(self.points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        self.points = self.points[:, not_close]

    def translate(self, x):
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix):
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix):
        self.points[:3, :] = transf_matrix.dot(np.vstack((self.points[:3, :], np.ones(self.nbr_points()))))[:3, :]

    def convertToPytorch(self):
        return torch.from_numpy(self.points)

    @staticmethod
    def fromPytorch(cls, pytorchTensor):
        points = pytorchTensor.numpy()
        return cls(points)

    def normalize(self, wlh):
        normalizer = [wlh[1], wlh[0], wlh[2]]
        self.points = self.points / np.atleast_2d(normalizer).T


class Box:
    def __init__(self, center, size, orientation, label=np.nan, score=np.nan, velocity=(np.nan, np.nan, np.nan),
                 name=None):
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}, name: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1], self.center[2], self.wlh[0],
                               self.wlh[1], self.wlh[2], self.orientation.axis[0], self.orientation.axis[1],
                               self.orientation.axis[2], self.orientation.degrees, self.orientation.radians,
                               self.velocity[0], self.velocity[1], self.velocity[2], self.name)

    def encode(self):
        return self.center.tolist() + self.wlh.tolist() + self.orientation.elements.tolist() + [self.label] + [
            self.score] + self.velocity.tolist() + [self.name]

    @classmethod
    def decode(cls, data):
        return Box(data[0:3], data[3:6], Quaternion(data[6:10]), label=data[10], score=data[11], velocity=data[12:15],
                   name=data[15])

    @property
    def rotation_matrix(self):
        return self.orientation.rotation_matrix

    def translate(self, x):
        self.center += x

    def rotate(self, quaternion):
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def transform(self, transf_matrix):
        transformed = np.dot(transf_matrix[0:3, 0:4].T, self.center)
        self.center = transformed[0:3] / transformed[3]
        self.orientation = self.orientation * Quaternion(matrix=transf_matrix[0:3, 0:3])
        self.velocity = np.dot(transf_matrix[0:3, 0:3], self.velocity)

    def corners(self, wlh_factor=1.0):
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self):
        return self.corners()[:, [2, 3, 7, 6]]


class SearchSpace(object):
    def reset(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def addData(self, data, score):
        return


class KalmanFiltering(SearchSpace):
    def __init__(self, bnd=None):
        self.bnd = [1, 1, 10] if bnd is None else bnd
        self.reset()

    def sample(self, n=10):
        return np.random.multivariate_normal(self.mean, self.cov, size=n)

    def addData(self, data, score):
        score = score.clip(min=1e-5)  # prevent sum=0 in case of bad scores
        self.data = np.concatenate((self.data, data))
        self.score = np.concatenate((self.score, score))
        self.mean = np.average(self.data, weights=self.score, axis=0)
        self.cov = np.cov(self.data.T, ddof=0, aweights=self.score)

    def reset(self):
        self.mean = np.zeros(len(self.bnd))
        self.cov = np.diag(self.bnd)
        if len(self.bnd) == 2:
            self.data = np.array([[], []]).T
        else:
            self.data = np.array([[], [], []]).T
        self.score = np.array([])


def get_box_by_offset(box: Box, offset, use_z=False):
    """
    offset:
        when called by dataset.py:
            x, y, z & theta(they use the same offset value), theta is in degree by default
        when called by eval.py:
            x, y, z, theta (theta is in degree by default)
    """
    rot_quat = Quaternion(matrix=box.rotation_matrix)
    trans = np.array(box.center)
    new_box = copy.deepcopy(box)
    new_box.translate(-trans)
    new_box.rotate(rot_quat.inverse)

    angle_in_radian = offset[-1] * np.pi / 180
    new_box.rotate(Quaternion(axis=[0, 0, 1], angle=angle_in_radian))   # angle means radian
    if offset[0] > new_box.wlh[0]:
        offset[0] = np.random.uniform(-1, 1)
    if offset[1] > min(new_box.wlh[1], 2):
        offset[1] = np.random.uniform(-1, 1)

    new_box.translate(np.array([offset[0], offset[1], offset[2] if use_z else 0]))
    new_box.rotate(rot_quat)
    new_box.translate(trans)
    return new_box


def get_model(PCs, boxes, offset=0., scale=1.0, normalize=False, visual_handle=None):
    if len(PCs) == 0:
        return PointCloud(np.ones((3, 0)))
    points = np.ones((PCs[0].points.shape[0], 0))

    for PC, box in zip(PCs, boxes):
        cropped_pc = crop_center_pc(PC, box, offset=offset, scale=scale, normalize=normalize)
        if cropped_pc.points.shape[1] > 0:
            points = np.concatenate([points, cropped_pc.points], axis=1)
        if visual_handle:
            print('[template]: model pc')
            visual_handle(PC.points.T, box=box)
    PC = PointCloud(points)
    if visual_handle:
        print('[template]: model pc')
        visual_handle(PC.points.T)
    return PC


def get_label_by_box(pc: PointCloud, box: Box, offset=0., scale=1.0):
    box_tmp = copy.deepcopy(box)
    new_pc = PointCloud(pc.points.copy())
    rot_mat = np.transpose(box_tmp.rotation_matrix)
    trans = -box_tmp.center

    new_pc.translate(trans)
    box_tmp.translate(trans)
    new_pc.rotate((rot_mat))
    box_tmp.rotate(Quaternion(matrix=(rot_mat)))

    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    # TODO
    try:
        x_filt_max = new_pc.points[0, :] < maxi[0]
        x_filt_min = new_pc.points[0, :] > mini[0]
        y_filt_max = new_pc.points[1, :] < maxi[1]
        y_filt_min = new_pc.points[1, :] > mini[1]
        z_filt_max = new_pc.points[2, :] < maxi[2]
        z_filt_min = new_pc.points[2, :] > mini[2]

        close = np.logical_and(x_filt_min, x_filt_max)
        close = np.logical_and(close, y_filt_min)
        close = np.logical_and(close, y_filt_max)
        close = np.logical_and(close, z_filt_min)
        close = np.logical_and(close, z_filt_max)
    except:
        import ipdb; ipdb.set_trace()

    new_label = np.zeros(new_pc.points.shape[1])
    new_label[close] = 1
    return new_label


def crop_pc(pc: PointCloud, box: Box, label=None, offset=0, scale=1.0):
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = pc.points[0, :] < maxi[0]
    x_filt_min = pc.points[0, :] > mini[0]
    y_filt_max = pc.points[1, :] < maxi[1]
    y_filt_min = pc.points[1, :] > mini[1]
    z_filt_max = pc.points[2, :] < maxi[2]
    z_filt_min = pc.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_pc = PointCloud(pc.points[:, close])
    if label is not None:
        label = label[close]
    return new_pc if label is None else (new_pc, label)


def crop_center_pc(pc, sample_box, gt_box=None, sample_offsets=None, offset=0.0, scale=1.0,
                   normalize=False, visual_handle=None, refine_box=True):
    new_pc = crop_pc(pc, sample_box, offset=2 * offset, scale=4 * scale)
    new_box = copy.deepcopy(sample_box)

    new_label = new_box_gt = label_reg = None
    if gt_box:
        new_label = get_label_by_box(new_pc, gt_box,
                                     offset=offset if refine_box else 0.0,
                                     scale=scale if refine_box else 1.0)
        new_box_gt = copy.deepcopy(gt_box)

    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center
    new_pc.translate(trans)
    new_box.translate(trans)
    new_pc.rotate(rot_mat)
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    if gt_box:
        new_pc, new_label = crop_pc(new_pc, new_box, new_label, offset=offset + gt_box.wlh[1] * 0.6, scale=1 * scale)
        new_box_gt.translate(trans)
        new_box_gt.rotate(Quaternion(matrix=(rot_mat)))
        try:
            label_reg = [new_box_gt.center[0], new_box_gt.center[1], new_box_gt.center[2], -sample_offsets[-1]]
            label_reg = np.array(label_reg)
        except:
            pass
    else:
        new_pc = crop_pc(new_pc, new_box, offset=offset, scale=scale)

    if visual_handle:
        print('[search]: crop pc and sample box')
        visual_handle(new_pc.points.T, box=new_box)
        print('[search]: crop pc and gt box')
        visual_handle(new_pc.points.T, box=new_box_gt)

    if normalize:
        new_pc.normalize(sample_box.wlh)
    return new_pc if not gt_box else (new_pc, new_label, label_reg)


def regularize_pc(pc: PointCloud, input_size, ratio=1, label=None, reg=None, istrain=True):
    if input_size > 0:
        input_size //= ratio
        pc = np.array(pc.points, dtype=np.float32)
        pc_dim = pc.shape[0]
        if np.shape(pc)[1] > 2:
            if pc.shape[1] != int(input_size):
                if not istrain:
                    common_utils.set_manual_seed(1)
                new_pts_idx = np.random.randint(
                    low=0, high=pc.shape[1], size=int(input_size), dtype=np.int64)
                pc = pc[:, new_pts_idx]
                if label is not None:
                    label = label[new_pts_idx]
            pc = pc.reshape((pc_dim, int(input_size))).T
            # if reg is not None:
            #     reg = np.tile(reg, [input_size // ratio, 1])
        else:
            pc = np.zeros((pc_dim, int(input_size))).T
            if label is not None:
                label = np.zeros(input_size)
            # if reg is not None:
            #     reg = np.tile(reg, [input_size // ratio, 1])
    else:
        pc = np.array(pc.points, dtype=np.float32)
    return pc if label is None else (pc, label, reg)
