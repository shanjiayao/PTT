# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ''' Pointnet2 layers.
# Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch
# Extended with the following:
# 1. Uniform sampling in each local region (sample_uniformly)
# 2. Return sampled points indices to support votenet.
# '''
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from . import pointnet2_utils as pointnet2_utils
from . import pytorch_utils as pt_utils
from ....utils.common_utils import square_distance


class PointnetSAModuleVotes(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            mlp: List[int],
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            normalize_xyz: bool = False,  # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            sample_method='fps'    # fps/rs/ffps
    ):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.normalize_xyz = normalize_xyz

        self.grouper = pointnet2_utils.QueryAndGroup(self.radius, nsample,
                                                     use_xyz=use_xyz, ret_grouped_xyz=True,
                                                     normalize_xyz=normalize_xyz,
                                                     sample_uniformly=sample_uniformly,
                                                     ret_unique_cnt=False)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)
        self.sample_method = sample_method

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor,
                npoint: int,
                inds: torch.Tensor = None):

        xyz_flipped = xyz.transpose(1, 2).contiguous()  # b c n
        if inds is None:
            if self.sample_method == 'ffps':
                features_for_fps = torch.cat([xyz_flipped, features], dim=1).transpose(1, 2).contiguous()    # b c n
                features_for_fps_distance = square_distance(features_for_fps, features_for_fps)
                inds = pointnet2_utils.furthest_point_sampling_with_dist(features_for_fps_distance, npoint)
            elif self.sample_method == 'rs':
                inds = torch.arange(npoint).repeat(xyz.size(0), 1).int().cuda()
            elif self.sample_method == 'sequence':
                    inds = torch.arange(npoint).repeat(xyz.size(0), 1).int().cuda()
            elif self.sample_method == 'fps':
                inds = pointnet2_utils.furthest_point_sample(xyz, npoint)
            else:
                raise NotImplementedError
        else:
            assert (inds.shape[1] == npoint)
            
        new_xyz = pointnet2_utils.gather_operation(
            xyz_flipped, inds
        ).transpose(1, 2).contiguous() if npoint is not None else None

        grouped_features, grouped_xyz = self.grouper(xyz, new_xyz, features)  # (B, C, npoint, nsample)
        new_features_ = self.mlp_module(grouped_features)  # (B, mlp[-1], npoint, nsample)
        new_features_ = F.max_pool2d(
            new_features_, kernel_size=[1, new_features_.size(3)]
        )  # (B, mlp[-1], npoint, 1)
        new_features_ = new_features_.squeeze(-1)  # (B, mlp[-1], npoint)

        return new_xyz, new_features_, inds.to(torch.int64)

