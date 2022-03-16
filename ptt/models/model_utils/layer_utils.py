# -*- coding: utf-8 -*-
import copy
import torch
from typing import List
import torch.nn as nn


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def make_fc_layers(input_channels, output_channels, fc_list):
    fc_layers = []
    pre_channel = input_channels
    for k in range(1, fc_list.__len__() - 1):
        fc_layers.extend([
            nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
            nn.BatchNorm1d(fc_list[k]),
            nn.ReLU(inplace=True)
        ])
        pre_channel = fc_list[k]
    fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
    fc_layers = nn.Sequential(*fc_layers)
    return fc_layers


def make_shared_mlp(mlp: List):
    shared_mlps = []
    for k in range(len(mlp) - 1):
        shared_mlps.extend([
            nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
            nn.BatchNorm2d(mlp[k + 1]),
            nn.ReLU(inplace=True)
        ])
    mlp_module = nn.Sequential(*shared_mlps)
    return mlp_module
