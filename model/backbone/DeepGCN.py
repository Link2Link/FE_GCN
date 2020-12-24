# _*_ coding: utf-8 _*_
# @Time : 2020/12/24
# @Author : Chenfei Wang
# @File : DeepGCN.py
# @desc :
# @note :

from pcdet.models.detectors.detector3d_template import Detector3DTemplate
import torch
from gcn_lib.dense import BasicConv, GraphConv2d, ResDynBlock2d, DenseDilatedKnnGraph, ResStaBlock2d
from torch.nn import Sequential as Seq
from model.sampler.FPS import SamplerHead
from model import sampler
import numpy as np

data_collect = []

class DeepGCN_Sta(torch.nn.Module):
    def __init__(self, cfg):
        super(DeepGCN_Sta, self).__init__()
        model_cfg = cfg.BACKBONE_3D
        act = 'relu'
        norm = 'batch'
        conv = 'edge'
        bias = True
        epsilon = 0.2
        stochastic = False

        self.fusion = model_cfg.FUSION
        self.in_channels = model_cfg.c_in + 3
        self.n_blocks = model_cfg.N_BLOCKS
        self.n_filters = model_cfg.N_FILTERS
        self.k = model_cfg.K
        self.diss = model_cfg.DISS
        assert len(self.n_blocks) == len(self.n_filters)

        self.num_point_features = model_cfg.FEATURE_OUT
        channels = [self.in_channels] + self.n_filters

        self.knn = DenseDilatedKnnGraph(self.k, 1, stochastic, epsilon)

        backbone_list = []
        self.feature_count = 0
        for i in range(len(channels) - 1):
            in_c = channels[i]
            out_c = channels[i+1]
            res_block_num = self.n_blocks[i] - 1
            backbone_list += [GraphConv2d(in_c, out_c, conv, act, norm, bias, diss=self.diss)]
            backbone_list += [ResStaBlock2d(out_c, conv, act, norm, bias, diss=self.diss) for i in range(res_block_num)]
            self.feature_count += out_c + out_c * (res_block_num)
        self.backbone = Seq(*backbone_list)

        if self.fusion:
            self.num_features = self.feature_count
        else:
            self.num_features = self.n_filters[-1]

        self.out_block = BasicConv([self.num_features, self.num_point_features], act, norm, bias)

    def forward(self, batch_dict):
        features = batch_dict['point_features']
        point_coords = batch_dict['point_coords']
        batch_size = batch_dict['batch_size']
        batch_indices = point_coords[:, 0].long()

        inputs_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            input = torch.cat([point_coords[bs_mask, 1:], features[bs_mask]], dim=1).transpose(0,1).unsqueeze(-1)
            inputs_list.append(input)
        inputs = torch.stack(inputs_list, dim=0)
        feat = inputs
        pos = inputs[:, 0:3]
        feature = inputs[:, 3:]
        topo = self.knn(pos)
        feat_list = []
        for i in range(len(self.backbone)):
            feat, _, _ = self.backbone[i](feat, topo, pos)
            feat_list.append(feat)

        if self.fusion:
            feats = torch.cat(feat_list, dim=1)
            feats = self.out_block(feats)
        else:
            feats = self.out_block(feat_list[-1])
        feats = feats.squeeze(-1).transpose(1, 2)
        point_features = torch.cat([f for f in feats])
        batch_dict['point_features'] = point_features
        batch_dict['point_coords'] = point_coords
        return batch_dict

class DeepGCN_Dyn(torch.nn.Module):
    def __init__(self, n_blocks=7, n_filters=16, k=20, act = 'relu', norm = 'batch', bias = True, epsilon = 0.2, stochastic = False, conv = 'edge'):
        super(DeepGCN_Dyn, self).__init__()
        in_channels = 4
        self.n_blocks = n_blocks
        self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(in_channels, n_filters, conv, act, norm, bias)
        self.backbone = Seq(*[ResDynBlock2d(n_filters, k, 1 + i, conv, act, norm, bias, stochastic, epsilon)
                              for i in range(self.n_blocks - 1)])
        self.num_point_features = n_filters


    def forward(self, inputs):
        topo_list = []
        topo = self.knn(inputs[:, 0:3])
        topo_list.append(topo.cpu().numpy()[0])
        feat = self.head(inputs, topo)
        for i in range(self.n_blocks - 1):
            feat, edge_index = self.backbone[i](feat)
            topo_list.append(edge_index.cpu().numpy()[0])
        return feat.squeeze(-1).transpose(1,2), topo_list