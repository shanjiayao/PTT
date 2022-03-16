# -*- coding: utf-8 -*-
import torch

from .voting_head_template import VotingHeadTemplate
from ..backbones_3d.pointnet2 import pointnet2_modules
from ..transformer_block import build_transformer
from ..backbones_3d.pointnet2 import pytorch_utils as layer_utils


class BoxVotingHead(VotingHeadTemplate):
    r"""Voting to obtain the centroid points from seed points. The score and votes generated from each seeds.
    """

    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg)
        self.vote_aggregation = pointnet2_modules.PointnetSAModuleVotes(
            radius=self.model_cfg.SA_CONFIG.RADIUS,
            nsample=self.model_cfg.SA_CONFIG.NSAMPLE,
            mlp=self.model_cfg.SA_CONFIG.MLPS,
            use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
            normalize_xyz=self.model_cfg.SA_CONFIG.get('NORMALIZE_XYZ', True),
            sample_method=self.model_cfg.SA_CONFIG.SAMPLE_METHOD
        )
        self.refine_layer = (
            layer_utils.Seq(self.model_cfg.FC[0])
                .conv1d(self.model_cfg.FC[1], bn=True)
                .conv1d(self.model_cfg.FC[2], bn=True)
                .conv1d(self.model_cfg.FC[3], activation=None)
        )
        if self.model_cfg.TRANSFORMER_BLOCK.ENABLE:
            self.transformer_block = build_transformer(self.model_cfg.TRANSFORMER_BLOCK)

    def get_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        mask = forward_ret_dict['mask']
        cls_pred = forward_ret_dict['pred_boxes_cls']
        cls_labels = forward_ret_dict['cls_label']

        boxes_cls_loss = self.cls_loss_func(cls_pred, cls_labels)
        boxes_cls_loss = torch.sum(boxes_cls_loss * mask) / (torch.sum(mask) + 1e-6)
        tb_dict = {'boxes_cls_loss': boxes_cls_loss.item()}
        boxes_cls_loss = boxes_cls_loss.float() * loss_cfgs.LOSS_WEIGHTS['boxes_cls_weight']
        return boxes_cls_loss, tb_dict

    def get_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        mask = forward_ret_dict['cls_label']
        reg_pred = forward_ret_dict['pred_boxes_reg']
        reg_labels = forward_ret_dict['reg_label']

        boxes_reg_loss = self.reg_loss_func(reg_pred, reg_labels[:, None, :].expand_as(reg_pred))
        boxes_reg_loss = (boxes_reg_loss.mean(2) * mask).sum() / (mask.sum() + 1e-06)
        tb_dict = {'boxes_reg_loss': boxes_reg_loss.item()}
        boxes_reg_loss = boxes_reg_loss.float() * loss_cfgs.LOSS_WEIGHTS['boxes_reg_weight']
        return boxes_reg_loss, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        boxes_loss_cls, tb_dict_1 = self.get_cls_layer_loss(self.forward_ret_dict)
        boxes_loss_reg, tb_dict_2 = self.get_reg_layer_loss(self.forward_ret_dict)

        boxes_loss = boxes_loss_cls + boxes_loss_reg
        boxes_loss = boxes_loss.float()
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        return boxes_loss, tb_dict

    def forward(self, batch_dict):
        if self.model_cfg.DEBUG:
            import ipdb; ipdb.set_trace()

        votes_feats = batch_dict['votes_feats']
        votes_coords = batch_dict['pred_centroids_votes']

        center_xyzs, proposal_feats, _ = self.vote_aggregation(
            xyz=votes_coords,
            features=votes_feats,
            npoint=self.model_cfg.SA_CONFIG.NPOINTS
        )

        if hasattr(self, 'transformer_block'):
            trans_feat = self.transformer_block(
                xyz=center_xyzs,
                features=proposal_feats.transpose(1, 2).contiguous()
            )[0]
            proposal_feats = trans_feat.transpose(1, 2).contiguous()

        proposal_offsets = self.refine_layer(proposal_feats)

        estimation_boxes = torch.cat((proposal_offsets[:, 0:3, :] + center_xyzs.transpose(1, 2).contiguous(),
                                     proposal_offsets[:, 3:, :]), dim=1)

        batch_dict['pred_box_center'] = center_xyzs
        batch_dict['pred_box_data'] = estimation_boxes.transpose(1, 2).contiguous()

        if self.training:
            dist = torch.sum((center_xyzs - batch_dict['reg_label'][:, None, 0:3]) ** 2, dim=-1)
            dist = torch.sqrt(dist + 1e-6)
            objectness_label = torch.zeros_like(dist, dtype=torch.float)
            objectness_mask = torch.zeros_like(objectness_label, dtype=torch.float)
            objectness_label[dist < 0.3] = 1
            objectness_mask[dist < 0.3] = 1
            objectness_mask[dist > 0.6] = 1
            self.forward_ret_dict = {
                'pred_boxes_cls': batch_dict['pred_box_data'][:, :, -1],
                'pred_boxes_reg': batch_dict['pred_box_data'][:, :, :-1],
                'mask': objectness_mask,
                'cls_label': objectness_label,
                'reg_label': batch_dict['reg_label']
            }

        return batch_dict
