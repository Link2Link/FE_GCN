# _*_ coding: utf-8 _*_
# @Time : 2020/12/25
# @Author : Chenfei Wang
# @File : DeepGCN_Dyn.py
# @desc :
# @note :

import torch
from .gcn_lib import BasicConv, GraphConv2d, DenseDilatedKnnGraph, ResStaBlock2d, TopoGraph, batched_index_select
from torch.nn import Sequential as Seq



class DeepGCN_Dyn(torch.nn.Module):
    def __init__(self, cfg):
        super(DeepGCN_Dyn, self).__init__()
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
            out_c = channels[i + 1]
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

        pos = inputs[:, 0:3]
        topo = self.knn(pos)
        graph = TopoGraph(x=inputs, edge_index=topo, pos=pos)   # the object graph used to transfor the imformation of graph
        feat_list = []
        for i in range(len(self.backbone)):
            feature, graph = self.backbone[i](graph)

            # update graph
            graph.x = feature
            topo = self.knn(feature)
            graph.edge_index = topo
            feat_list.append(feature)

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