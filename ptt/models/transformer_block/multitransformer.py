# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from thop import clever_format
from ..model_utils import square_distance, index_points, get_clones


class MulHeadTransformerLayer(nn.Module):
    def __init__(self, d_points, d_model, k, heads, drop=0.) -> None:
        super().__init__()
        self.heads = heads
        head_dim = d_model // heads
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj_drop = nn.Dropout(drop)
        self.k = k
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_points)

    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)  # 2ms
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        B, N, C = x.shape
        query, key, value = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        query = query.view(B, N, self.heads, -1).permute(0, 2, 1, 3).flatten(0, 1)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        pos_enc, key, value = map(
            lambda t: t.view(B, N, t.shape[2], self.heads, -1).permute(0, 3, 1, 2, 4).flatten(0, 1), \
            (pos_enc, key, value))

        attn = self.fc_gamma(query[:, :, None] - key + pos_enc)
        attn = F.softmax(attn / np.sqrt(key.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, value + pos_enc)
        if self.heads > 1:
            res = res.permute(0, 2, 1).reshape(B, C, N).permute(0, 2, 1)

        res = self.norm1(self.proj_drop(self.proj(res)))
        res = self.norm2(self.fc2(res)) + pre
        return res, attn


class MulTransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k, heads, layers):
        super().__init__()
        transformer_layer = MulHeadTransformerLayer(d_points, d_model, k, heads)
        self.layers = get_clones(transformer_layer, layers)

    def forward(self, xyz, features):
        output = features
        for layer in self.layers:
            output, attn = layer(xyz, output)
        return output, attn


if __name__ == "__main__":
    nhead = 8
    nlayer = 8
    block = MulTransformerBlock(d_points=512, d_model=512, k=2, heads=nhead, layers=nlayer)
    xyz = torch.randn(2, 5, 3)
    feats = torch.randn(2, 5, 512)
    o = block(xyz, feats)[0]
    flops, params = profile(block, inputs=(xyz, feats,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print('params: ', params)
    print('flops: ', flops)
    print('nhead: ', nhead)
    print('nlayer: ', nlayer)
