# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones_3d.pointnet2 import pytorch_utils as layer_utils


class CosineSimAug(nn.Module):
    r"""Calculate the cosine similarity between two features. Then embed the template info into the search feature
        from P2B: https://github.com/HaozheQi/P2B/blob/master/pointnet2/models/pointnet_tracking.py#L136
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.cosine = nn.CosineSimilarity(dim=1)
        self.mlp = layer_utils.SharedMLP(self.model_cfg.MLP.CHANNELS, bn=self.model_cfg.MLP.BN)
        self.conv = (
            layer_utils.Seq(self.model_cfg.CONV.CHANNELS[0])
                .conv1d(self.model_cfg.CONV.CHANNELS[1], bn=self.model_cfg.CONV.BN)
                .conv1d(self.model_cfg.CONV.CHANNELS[2], activation=None)
        )

    def forward(self, batch_dict):
        if self.model_cfg.DEBUG:
            import ipdb; ipdb.set_trace()

        search_feats = batch_dict['search_feats']
        template_feats = batch_dict['template_feats']
        template_xyz = batch_dict['template_seeds']
        b, _, _ = search_feats.shape

        f, n1, n2 = search_feats.shape[1], template_feats.shape[-1], search_feats.shape[-1]
        sim_feat = self.cosine(template_feats.unsqueeze(-1).expand(b, f, n1, n2),
                               search_feats.unsqueeze(2).expand(b, f, n1, n2))
        template_xyz_ = template_xyz.transpose(1, 2).contiguous().unsqueeze(-1).expand(b, 3, n1, n2)
        fusion_feature = torch.cat((sim_feat.unsqueeze(1), template_xyz_), dim=1)
        fusion_feature = torch.cat((fusion_feature, template_feats.unsqueeze(-1).expand(b, f, n1, n2)), dim=1)
        fusion_feature = self.mlp(fusion_feature)
        fusion_feature = F.max_pool2d(fusion_feature, kernel_size=[fusion_feature.size(2), 1])
        fusion_feature = fusion_feature.squeeze(2)
        fusion_feature = self.conv(fusion_feature)  # b c n

        batch_dict['cosine_feats'] = fusion_feature
        return batch_dict
