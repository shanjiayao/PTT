import torch
import torch.nn as nn

from typing import List
from .pointnet2 import pointnet2_modules


class PointNet2BackboneLight(nn.Module):
    """Modified based on code of P2B,
    ref to https://github.com/HaozheQi/P2B/blob/master/pointnet2/models/pointnet_tracking.py#L18"""
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        input_channels -= 3
        self.num_points_each_layer = []
        self.SA_modules = nn.ModuleList()

        for k in range(self.model_cfg.SA_CONFIG.RADIUS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            mlps[0] = input_channels if k == 0 else mlps[0]
            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleVotes(
                    radius=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsample=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                    normalize_xyz=self.model_cfg.SA_CONFIG.get('NORMALIZE_XYZ', True),
                    sample_method=self.model_cfg.SA_CONFIG.SAMPLE_METHOD[k]
                )
            )

        self.cov_final = nn.Conv1d(256, 256, kernel_size=1)
        self.num_point_features = self.model_cfg.SA_CONFIG.MLPS[-1][-1]

    @staticmethod
    def _break_up_pc(pc):
        xyz = pc[..., 0:3].contiguous()
        features = (pc[..., 3:].contiguous() if pc.size(-1) > 3 else None)
        return xyz, features

    def branch_forward(self, pts, npoints: List):
        xyz, features = self._break_up_pc(pts)  # n, 1 / n, 3
        xyz, features, inds0 = self.SA_modules[0](xyz=xyz, features=features, npoint=npoints[0])
        xyz, features, inds1 = self.SA_modules[1](xyz=xyz, features=features, npoint=npoints[1])
        xyz, features, inds2 = self.SA_modules[2](xyz=xyz, features=features, npoint=npoints[2])
        point_features = self.cov_final(features)
        assert inds1.dtype == inds2.dtype == torch.int64, 'index type must be int64, not {}'.format(inds2.dtype)
        inds = inds0.gather(1, inds1).gather(1, inds2)  # b

        return xyz, point_features, inds

    def forward(self, batch_dict):
        if self.model_cfg.DEBUG:
            import ipdb; ipdb.set_trace()

        batch_dict['search_seeds'], batch_dict['search_feats'], batch_dict['search_inds'] = self.branch_forward(
            batch_dict['search_points'], self.model_cfg.SA_CONFIG.NPOINTS_SEARCH
        )

        batch_dict['template_seeds'], batch_dict['template_feats'], batch_dict['template_inds'] = self.branch_forward(
            batch_dict['template_points'], self.model_cfg.SA_CONFIG.NPOINTS_TEMPLATE
        )

        batch_dict.pop('search_points')
        batch_dict.pop('template_points')

        return batch_dict
