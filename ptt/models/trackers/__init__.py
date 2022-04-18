# -*- coding: utf-8 -*-
from .ptt import PTT
#from .p2b import P2B

__all__ = {
    'PTT': PTT,
    #'P2B': P2B
}


def build_tracker(model_cfg, num_class, dataset):
    return __all__[model_cfg.NAME](
        model_cfg=model_cfg,
        num_class=num_class,
        dataset=dataset
    )
