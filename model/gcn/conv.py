import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv1d
from .tools import *

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
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer

class BasicConv(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv1d(channels[i - 1], channels[i], 1, bias=bias))
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

class EdgeConv(torch.nn.Module):
    """
    forward input : x -> [N, F, 1]
                    index -> [N, K]
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, diss=True):
        super(EdgeConv, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)
        self.diss = diss
        self.linear = BasicConv([out_channels, int(out_channels/4)], act=None, norm=None, bias=True)
        self.linear2 = BasicConv([out_channels, int(out_channels/4)], act=None, norm=None, bias=True)

    def forward(self, x, index, pos):

        num_points, k = index.shape
        x_i = x.repeat(1, 1, k)
        x_j = index_select(x, index)
        feature = self.nn(torch.cat([x_i, x_j - x_i], dim=1))
        K = self.linear(feature)
        Q = self.linear2(feature)

        r_matix = torch.matmul(K.transpose(1,2), Q)
        att = torch.softmax(r_matix, dim=1)
        feature = torch.matmul(feature, att)
        if self.diss:
            pos_i = pos.repeat(1, 1, k)
            pos_j = index_select(pos, index)
            vec = pos_j - pos_i
            dis = torch.sqrt(torch.sum(torch.square(vec), dim=1, keepdim=True))
            scale = 2 * torch.sigmoid(-dis)
            max_value, _ = torch.max(scale*feature, -1, keepdim=True)
        else:
            max_value, _ = torch.max(feature, -1, keepdim=True)

        return max_value




