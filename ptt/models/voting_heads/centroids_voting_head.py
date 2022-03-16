# -*- coding: utf-8 -*-
import torch

from .voting_head_template import VotingHeadTemplate
from ..backbones_3d.pointnet2 import pytorch_utils as layer_utils
from ..transformer_block import build_transformer


class CentroidVotingHead(VotingHeadTemplate):
    r"""Voting to obtain the centroid points from seed points. The score and votes generated from each seeds.
    """
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg)
        self.cla_layer = (
            layer_utils.Seq(self.model_cfg.CLS_FC.CHANNELS[0])
                .conv1d(self.model_cfg.CLS_FC.CHANNELS[1], bn=True)
                .conv1d(self.model_cfg.CLS_FC.CHANNELS[2], bn=True)
                .conv1d(self.model_cfg.CLS_FC.CHANNELS[3], activation=None)
        )
        self.vote_layer = (
            layer_utils.Seq(self.model_cfg.REG_FC.CHANNELS[0])
                .conv1d(self.model_cfg.REG_FC.CHANNELS[1], bn=True)
                .conv1d(self.model_cfg.REG_FC.CHANNELS[2], bn=True)
                .conv1d(self.model_cfg.REG_FC.CHANNELS[3], activation=None)
        )
        if self.model_cfg.TRANSFORMER_BLOCK.ENABLE:
            self.transformer_block = build_transformer(self.model_cfg.TRANSFORMER_BLOCK)

    def get_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        cls_pred = forward_ret_dict['pred_centroids_cls'].view(-1)
        cls_labels = forward_ret_dict['cls_label'].view(-1)

        centroids_cls_loss = self.cls_loss_func(cls_pred, cls_labels)

        tb_dict = {'centroids_cls_loss': centroids_cls_loss.item()}
        centroids_cls_loss = centroids_cls_loss.float() * loss_cfgs.LOSS_WEIGHTS['centroids_cls_weight']
        return centroids_cls_loss, tb_dict

    def get_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        mask = forward_ret_dict['cls_label']
        reg_pred = forward_ret_dict['pred_centroids_votes']
        reg_labels = forward_ret_dict['reg_label']
        # import ipdb; ipdb.set_trace()
        centroids_reg_loss = self.reg_loss_func(reg_pred, reg_labels[:, None, :3].expand_as(reg_pred))
        centroids_reg_loss = (centroids_reg_loss.mean(2) * mask).sum() / (mask.sum() + 1e-06)

        tb_dict = {'centroids_reg_loss': centroids_reg_loss.item()}
        centroids_reg_loss = centroids_reg_loss.float() * loss_cfgs.LOSS_WEIGHTS['centroids_reg_weight']
        return centroids_reg_loss, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        centroids_loss_cls, tb_dict_1 = self.get_cls_layer_loss(self.forward_ret_dict)
        centroids_loss_reg, tb_dict_2 = self.get_reg_layer_loss(self.forward_ret_dict)

        centroids_loss = centroids_loss_cls + centroids_loss_reg
        centroids_loss = centroids_loss.float()
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        return centroids_loss, tb_dict

    def forward(self, batch_dict):
        if self.model_cfg.DEBUG:
            import ipdb; ipdb.set_trace()

        search_seeds_xyz = batch_dict['search_seeds'].transpose(1, 2).contiguous()    # b 3 n
        fusion_feats = batch_dict['cosine_feats']   # b c n

        if hasattr(self, 'transformer_block'):
            trans_feat = self.transformer_block(
                xyz=batch_dict['search_seeds'],
                features=fusion_feats.transpose(1, 2).contiguous()
            )[0]
            fusion_feats = trans_feat.transpose(1, 2).contiguous()

        if hasattr(self.model_cfg, 'CLS_USE_SEARCH_XYZ') and self.model_cfg.CLS_USE_SEARCH_XYZ:
            fusion_feats = torch.cat((search_seeds_xyz, fusion_feats), dim=1)
            classifying_output = self.cla_layer(fusion_feats).squeeze(1)
            cls_pred = classifying_output.squeeze(0)
            classifying_score = classifying_output.sigmoid()

            voting_output = self.vote_layer(fusion_feats)
            voting_input = fusion_feats
        else:
            classifying_output = self.cla_layer(fusion_feats).squeeze(1)
            cls_pred = classifying_output.squeeze(0)
            classifying_score = classifying_output.sigmoid()

            voting_input = torch.cat((search_seeds_xyz, fusion_feats), dim=1)
            voting_output = self.vote_layer(voting_input)

        voting_results = voting_input + voting_output
        votes_coords = voting_results[:, 0:3, :].transpose(1, 2).contiguous()    # b n 3

        batch_dict['pred_centroids_cls'] = cls_pred  # b*n
        batch_dict['pred_centroids_votes'] = votes_coords  # b n 3
        batch_dict['votes_feats'] = torch.cat((classifying_score.unsqueeze(1), voting_results[:, 3:, :]), dim=1)

        if self.training:
            self.forward_ret_dict = {
                'pred_centroids_cls': batch_dict['pred_centroids_cls'],
                'pred_centroids_votes': batch_dict['pred_centroids_votes'],
                'cls_label': batch_dict['cls_label'].gather(1, batch_dict['search_inds']),
                'reg_label': batch_dict['reg_label']
            }

        return batch_dict
