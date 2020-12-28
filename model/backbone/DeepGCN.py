# _*_ coding: utf-8 _*_
# @Time : 2020/12/24
# @Author : Chenfei Wang
# @File : DeepGCN.py
# @desc :
# @note :

import torch
from .blocks import GraphConv, ResGraphConv, Graph
from torch.nn import Sequential as Seq

data_collect = []

class DeepGCN_Sta(torch.nn.Module):
    def __init__(self, cfg):
        super(DeepGCN_Sta, self).__init__()
        model_cfg = cfg.BACKBONE_3D

        self.fusion = model_cfg.FUSION
        self.in_channels = model_cfg.c_in + 3
        self.n_blocks = model_cfg.N_BLOCKS
        self.n_filters = model_cfg.N_FILTERS
        self.k = model_cfg.K
        self.diss = model_cfg.DISS
        self.att = model_cfg.ATT

        assert len(self.n_blocks) == len(self.n_filters)

        self.num_point_features = model_cfg.FEATURE_OUT
        channels = [self.in_channels] + self.n_filters

        # self.knn = DenseDilatedKnnGraph(self.k, 1, stochastic, epsilon)

        backbone_list = []
        self.feature_count = 0
        for i in range(len(channels) - 1):
            in_c = channels[i]
            out_c = channels[i+1]
            res_block_num = self.n_blocks[i] - 1
            backbone_list += [GraphConv(in_c, out_c, conv='linear', bias=True)]
            backbone_list += [ResGraphConv(out_c,
                                           conv='edge',
                                           act='relu',
                                           norm='batch',
                                           bias=True,
                                           DISS=self.diss,
                                           ATT=self.att) for i in range(res_block_num)]
            self.feature_count += out_c + out_c * (res_block_num)
        self.backbone = Seq(*backbone_list)

        if self.fusion:
            self.num_features = self.feature_count
        else:
            self.num_features = self.n_filters[-1]

        self.out_block = GraphConv(self.num_features, self.num_point_features, conv='linear', bias=True)

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
        # topo = self.knn(pos)
        graph = Graph(x=inputs, pos=pos, k=self.k)   # the object graph used to transfor the imformation of graph
        feat_list = []

        for i in range(len(self.backbone)):
            feature, graph = self.backbone[i](graph)

            # update graph
            graph.x = feature
            feat_list.append(feature)

        if self.fusion:
            graph.x = torch.cat(feat_list, dim=1)
            feature, graph = self.out_block(graph)
        else:
            graph.x = feat_list[-1]
            feature, graph = self.out_block(graph)
        feats = feature.squeeze(-1).transpose(1, 2)
        point_features = torch.cat([f for f in feats])
        batch_dict['point_features'] = point_features
        batch_dict['point_coords'] = point_coords
        return batch_dict

