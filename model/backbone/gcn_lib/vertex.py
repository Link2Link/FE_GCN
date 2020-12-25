import torch
from torch import nn
from .nn import BasicConv, batched_index_select
from .edge import DenseDilatedKnnGraph, DilatedKnnGraph
import torch.nn.functional as F
from .graph import TopoGraph

class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, diss=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)
        self.diss = diss

    def forward(self, graph:TopoGraph) -> (torch.Tensor, TopoGraph):
        x = graph.x
        edge_index = graph.edge_index
        pos = graph.pos

        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        feature = self.nn(torch.cat([x_i, x_j - x_i], dim=1))

        if self.diss:
            pos_i = batched_index_select(pos, edge_index[1])
            pos_j = batched_index_select(pos, edge_index[0])
            vec = pos_j - pos_i
            dis = torch.sqrt(torch.sum(torch.square(vec), dim=1))
            suppression = 2 * torch.sigmoid(-dis).unsqueeze(1)
            feature = feature * suppression
        max_value, _ = torch.max(feature, -1, keepdim=True)

        return max_value, graph

class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True, diss=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias, diss)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, graph:TopoGraph) -> (torch.Tensor, TopoGraph):
        return self.gconv(graph)


class ResStaBlock2d(nn.Module):
    def __init__(self, in_channels, conv='edge', act='relu', norm=None,
                 bias=True, res_scale=1, diss=True):
        super(ResStaBlock2d, self).__init__()
        self.body = GraphConv2d(in_channels, in_channels, conv, act, norm, bias, diss)
        self.res_scale = res_scale

    def forward(self, graph:TopoGraph) -> (torch.Tensor, TopoGraph):
        res, graph = self.body(graph)
        feature = self.res_scale * graph.x + res
        return feature, graph
