# _*_ coding: utf-8 _*_
# @Time : 2020/12/29
# @Author : Chenfei Wang
# @File : gcn.py
# @desc :
# @note :

import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d
from torch.nn import Sequential as Seq
from .data import *
from .gcn_basic import *

class ResGCN(nn.Module):
    def __init__(self, in_channels, blocks, DISS=False, ATT=False):
        super(ResGCN, self).__init__()
        model_list = [ResGraphConv(in_channels,
                                   conv='edge',
                                   act='relu',
                                   norm='batch',
                                   bias=True,
                                   DISS=DISS,
                                   ATT=ATT) for i in range(blocks)]
        self.model = Seq(*model_list)

    def forward(self, graph:Graph) -> Graph:
        for i in range(len(self.model)):
            feature, graph = self.model[i](graph)
            graph.x = feature
        return graph

class EncoderGCN(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, num_points, k, DISS=False, ATT=False):
        super(EncoderGCN, self).__init__()
        self.encoder = GraphEncoderBlock(in_channels, out_channels, num_points, k=k)
        self.inference = ResGCN(out_channels, blocks, DISS=DISS, ATT=ATT)

    def forward(self, graph: Graph):
        subgraph = self.encoder(graph)
        subgraph = self.inference(subgraph)
        return subgraph

class DecoderGCN(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, DISS=False, ATT=False, out=None):
        super(DecoderGCN, self).__init__()
        self.decoder = GraphUpSamplingLayer()
        if out is not None:
            self.feature_mix = GraphConv(in_channels + out_channels, out, conv='edge', bias=True, DISS=DISS, ATT=ATT)
            self.inference = ResGCN(out, blocks - 1, DISS=DISS, ATT=ATT)
        else:
            self.feature_mix = GraphConv(in_channels + out_channels, out_channels, conv='edge', bias=True, DISS=DISS, ATT=ATT)
            self.inference = ResGCN(out_channels, blocks - 1, DISS=DISS, ATT=ATT)


    def forward(self, subgraph: Graph, graph:Graph):
        feature_sub = self.decoder(subgraph, graph)
        feature = torch.cat([graph.x, feature_sub], dim=1)
        graph.x = feature
        feature, graph = self.feature_mix(graph)
        graph.x = feature
        graph = self.inference(graph)
        return graph



