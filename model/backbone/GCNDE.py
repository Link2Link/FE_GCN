# _*_ coding: utf-8 _*_
# @Time : 2020/12/29
# @Author : Chenfei Wang
# @File : GCNDE.py
# @desc :
# @note :

import torch
from .blocks import GraphConv, ResGraphConv, Graph, \
    GraphDownSamplingLayer, GraphUpSamplingLayer, \
    GraphPooing, ResGCN, GraphEncoderBlock, EncoderGCN, DecoderGCN

from torch.nn import Sequential as Seq

class DeepGCN_ED(torch.nn.Module):
    def __init__(self, cfg):
        super(DeepGCN_ED, self).__init__()
        model_cfg = cfg.BACKBONE_3D

        self.fusion = model_cfg.FUSION
        self.in_channels = model_cfg.c_in + 3
        self.n_blocks = model_cfg.N_BLOCKS
        self.n_filters = model_cfg.N_FILTERS
        self.k = model_cfg.K
        self.diss = model_cfg.DISS
        self.att = model_cfg.ATT
        self.en_struct = model_cfg.ENCODER
        assert len(self.n_blocks) == len(self.n_filters)
        assert len(self.n_blocks) == len(self.en_struct)
        self.num_point_features = model_cfg.FEATURE_OUT
        channels = [self.in_channels] + self.n_filters

        self.encoder1 = EncoderGCN(channels[0], channels[1], self.n_blocks[0], self.en_struct[0],
                                   self.k, DISS=self.diss, ATT=self.att)
        self.encoder2 = EncoderGCN(channels[1], channels[2], self.n_blocks[1], self.en_struct[1],
                                   self.k, DISS=self.diss, ATT=self.att)
        self.encoder3 = EncoderGCN(channels[2], channels[3], self.n_blocks[2], self.en_struct[2],
                                   self.k, DISS=self.diss, ATT=self.att)
        self.decoder3 = DecoderGCN(channels[3], channels[2], self.n_blocks[2], DISS=self.diss, ATT=self.att)
        self.decoder2 = DecoderGCN(channels[2], channels[1], self.n_blocks[1], DISS=self.diss, ATT=self.att)
        self.decoder1 = DecoderGCN(channels[1], channels[0], self.n_blocks[0], DISS=self.diss, ATT=self.att,
                                   out=self.num_point_features)

    def forward(self, batch_dict):
        features = batch_dict['point_features']
        point_coords = batch_dict['point_coords']
        batch_size = batch_dict['batch_size']
        batch_indices = point_coords[:, 0].long()

        inputs_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            input = torch.cat([point_coords[bs_mask, 1:], features[bs_mask]], dim=1).transpose(0, 1).unsqueeze(-1)
            inputs_list.append(input)
        inputs = torch.stack(inputs_list, dim=0)

        pos = inputs[:, 0:3]
        # topo = self.knn(pos)
        graph = Graph(x=inputs, pos=pos, k=self.k)  # the object graph used to transfor the imformation of graph


        # encoder
        subgraph1 = self.encoder1(graph)
        subgraph2 = self.encoder2(subgraph1)
        subgraph3 = self.encoder3(subgraph2)

        # decoder
        subgraph2 = self.decoder3(subgraph3, subgraph2)
        subgraph1 = self.decoder2(subgraph2, subgraph1)
        out_graph = self.decoder1(subgraph1, graph)

        feature = out_graph.x
        feats = feature.squeeze(-1).transpose(1, 2)
        point_features = torch.cat([f for f in feats])
        batch_dict['point_features'] = point_features
        batch_dict['point_coords'] = point_coords
        return batch_dict


