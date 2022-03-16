import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import common_utils


class VotingHeadTemplate(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = 1

        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([losses_cfg.CLS_LOSS_POS_WEIGHT], dtype=torch.float32).detach(),
                reduction=losses_cfg.CLS_LOSS_REDUCTION
            ).cuda())

        self.add_module('reg_loss_func', nn.SmoothL1Loss(reduction='none').cuda())
