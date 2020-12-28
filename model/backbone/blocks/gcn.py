
# _*_ coding: utf-8 _*_
# @Time : 2020/12/28
# @Author : Chenfei Wang
# @File : gcn.py
# @desc :
# @note :

import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d
from .data import *

"""
basic layers 
"""
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm, nc):
    # normalization layer 2d
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


class MLP(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
        super(MLP, self).__init__(*m)


class BasicConv(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicConv, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

##########################################################################################################
"""
basic layers for GCN
"""
class EdgeConv(nn.Module):
    """
    Edge convolution layer with activation, batch normalization, distance supression (DISS), attention (ATT)
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, DISS=True, ATT=True, k=20):
        super(EdgeConv, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)
        self.DISS = DISS
        self.ATT = ATT
        if self.DISS and self.ATT:
            self.att = Conv2d(k, 1, 1, bias=bias)

    def forward(self, graph:Graph) -> (torch.Tensor, Graph):
        # unpack graph data
        x = graph.x
        edge_index = graph.edge_index
        pos = graph.pos
        dis = graph.dis

        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        feature = self.nn(torch.cat([x_i, x_j - x_i], dim=1))

        if self.DISS:
            if self.ATT:
                scaler = self.att(dis.transpose(0, 2).unsqueeze(0)).squeeze(0).transpose(0, 2)
                scaler = torch.tanh(scaler) + 1
                suppression = 2 * torch.sigmoid(-dis * scaler).unsqueeze(1)
            else:
                suppression = 2 * torch.sigmoid(-dis).unsqueeze(1)
            feature = feature * suppression
        max_value, _ = torch.max(feature, -1, keepdim=True)
        return max_value, graph

class LinearConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(LinearConv, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act=None, norm=None, bias=bias)

    def forward(self, graph: Graph) -> (torch.Tensor, Graph):
        return self.nn(graph.x), graph


##########################################################################################################
"""
abstract layers for GCN
"""
class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """

    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True, DISS=True, ATT=True, k=20):
        super(GraphConv, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv(in_channels, out_channels, act, norm, bias, DISS=DISS, ATT=ATT, k=k)
        elif conv == 'linear':
            self.gconv = LinearConv(in_channels, out_channels, bias=bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, graph:Graph) -> (torch.Tensor, Graph):
        return self.gconv(graph)

class ResGraphConv(nn.Module):
    def __init__(self, in_channels, conv='edge', act='relu', norm=None,
                 bias=True, res_scale=1, DISS=True, ATT=True, k=20):
        super(ResGraphConv, self).__init__()
        self.body = GraphConv(in_channels, in_channels, conv, act, norm, bias, DISS=DISS, ATT=ATT, k=k)
        self.res_scale = res_scale

    def forward(self, graph:Graph) -> (torch.Tensor, Graph):
        res, graph = self.body(graph)
        feature = self.res_scale * graph.x + res
        return feature, graph













