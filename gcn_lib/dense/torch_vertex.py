import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select
from .torch_edge import DenseDilatedKnnGraph, DilatedKnnGraph
import torch.nn.functional as F


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        return self.nn(torch.cat([x, x_j], dim=1))

import time

class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, diss=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)
        self.diss = diss

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        feature = self.nn(torch.cat([x_i, x_j - x_i], dim=1))
        max_value, _ = torch.max(feature, -1, keepdim=True)
        return max_value

    def forward(self, x, edge_index, pos):
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
        return max_value, edge_index, pos


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True, diss=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias, diss)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index), edge_index

    def forward(self, x, edge_index, pos):
        return self.gconv(x, edge_index, pos)



class DynConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DynConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        if knn == 'matrix':
            self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        else:
            self.dilated_knn_graph = DilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x):
        edge_index = self.dilated_knn_graph(x)
        return super(DynConv2d, self).forward(x, edge_index), edge_index

    
class PlainDynBlock2d(nn.Module):
    """
    Plain Dynamic graph convolution block
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, knn='matrix'):
        super(PlainDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x):
        return self.body(x)
    
    
class ResDynBlock2d(nn.Module):
    """
    Residual Dynamic graph convolution block
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, knn='matrix', res_scale=1):
        super(ResDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)
        self.res_scale = res_scale

    def forward(self, x):
        res_body, edge_index = self.body(x)
        return res_body + x*self.res_scale, edge_index


class DenseDynBlock2d(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, in_channels, out_channels=64,  kernel_size=9, dilation=1, conv='edge',
                 act='relu', norm=None,bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DenseDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, out_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x):
        dense = self.body(x)
        return torch.cat((x, dense), 1)


class ResStaBlock2d(nn.Module):
    def __init__(self, in_channels, conv='edge', act='relu', norm=None,
                 bias=True, res_scale=1, diss=True):
        super(ResStaBlock2d, self).__init__()
        self.body = GraphConv2d(in_channels, in_channels, conv, act, norm, bias, diss)
        self.res_scale = res_scale

    def forward(self, x, edge_index):
        res_body, edge_index = self.body(x, edge_index)
        return res_body + x*self.res_scale, edge_index

    def forward(self, x, edge_index, pos):
        res_body, edge_index, pos = self.body(x, edge_index, pos)
        return res_body + x*self.res_scale, edge_index, pos
