import numpy as np

from ...utils import common_utils


def random_flip_along_x(data_dict):
    """
    Args:
        data_dict:
            'search_points': n, 3
            'template_points': n, 3
            'cls_label': n
            'reg_label': n, 4 [x, y, z, theta]
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        data_dict['search_points'][:, 1] = -data_dict['search_points'][:, 1]
        data_dict['template_points'][:, 1] = -data_dict['template_points'][:, 1]
        data_dict['reg_label'][:, 1] = -data_dict['reg_label'][:, 1]
        data_dict['reg_label'][:, -1] = -data_dict['reg_label'][:, -1]
    return data_dict


def random_flip_along_y(data_dict):
    """
    Args:
        data_dict:
            'search_points': n, 3
            'template_points': n, 3
            'cls_label': n
            'reg_label': n, 4 [x, y, z, theta]
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        data_dict['template_points'][:, 0] = -data_dict['template_points'][:, 0]
        data_dict['search_points'][:, 0] = -data_dict['search_points'][:, 0]
        data_dict['reg_label'][:, 0] = -data_dict['reg_label'][:, 0]
        data_dict['reg_label'][:, -1] = -(data_dict['reg_label'][:, -1] + np.pi)
    return data_dict


def global_rotation(data_dict, rot_range):
    """
    Args:
        data_dict:
            'search_points': n, 3
            'template_points': n, 3
            'cls_label': n
            'reg_label': n, 4 [x, y, z, theta]
        rot_range: random value
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    data_dict['search_points'] = common_utils.rotate_points_along_z(
        data_dict['search_points'][np.newaxis, :, :], np.array([noise_rotation])
    )[0]
    data_dict['template_points'] = common_utils.rotate_points_along_z(
        data_dict['template_points'][np.newaxis, :, :], np.array([noise_rotation])
    )[0]
    data_dict['reg_label'][:, 0:3] = common_utils.rotate_points_along_z(
        data_dict['reg_label'][np.newaxis, :, 0:3], np.array([noise_rotation])
    )[0]
    data_dict['reg_label'][:, -1] += noise_rotation
    return data_dict


def global_scaling(data_dict, scale_range):
    """
    Args:
        data_dict:
            'search_points': n, 3
            'template_points': n, 3
            'cls_label': n
            'reg_label': n, 4 [x, y, z, theta]
        scale_range: random value
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return data_dict
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    data_dict['search_points'][:, :3] *= noise_scale
    data_dict['template_points'][:, :3] *= noise_scale
    data_dict['reg_label'][:, :3] *= noise_scale
    return data_dict
