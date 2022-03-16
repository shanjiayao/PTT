from collections import namedtuple

import numpy as np
import torch

from .trackers import build_tracker


def build_network(model_cfg, num_class, dataset):
    return build_tracker(model_cfg=model_cfg, num_class=num_class, dataset=dataset)


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        try:
            if isinstance(val, np.ndarray):
                batch_dict[key] = torch.from_numpy(val).float().cuda()
            elif isinstance(val, torch.Tensor):
                batch_dict[key] = val.float().cuda()
        except:
            pass


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
