import torch.nn as nn
from ..builder import VOXEL_ENCODERS
from .voxel_encoder import DynamicVFE

import torch
from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import DynamicScatter
from .. import builder
# from ..registry import VOXEL_ENCODERS
from .utils import VFELayer, get_paddings_indicator

from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv1d
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn import ModuleList

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
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, diss=True, merge=True, att=True):
        super(EdgeConv, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)
        self.diss = diss
        self.linear = BasicConv([out_channels, 1 if merge else int(out_channels/4)], act=None, norm=None, bias=False)
        self.linear2 = BasicConv([out_channels, 1 if merge else int(out_channels/4)], act=None, norm=None, bias=False)
        self.ATT = att

    def forward(self, x, index, pos):

        num_points, k = index.shape
        x_i = x.repeat(1, 1, k)
        x_j = index_select(x, index)
        feature = self.nn(torch.cat([x_i, x_j - x_i], dim=1))
        if self.ATT:
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

def index_select(x, idx):
    """
    input :     x -> [N, F, 1]     F is feature dimension
                batch_idx -> [N]
                idx -> [M, K]
    output :    selected -> [M, F, K]
    """
    N, F = x.shape[:2]
    M, K = idx.shape
    idx = idx.contiguous().view(-1)
    selected = x.transpose(2, 1)[idx]
    selected = selected.view(M, K, F).permute(0, 2, 1).contiguous()
    return selected

def pointwise_distance(x, y, square=True):
    """
    The pointwise distance from x to y
    """
    with torch.no_grad():
        x = x.squeeze(-1)
        y = y.squeeze(-1)

        x = x.unsqueeze(-1)
        y = y.transpose(0,1).unsqueeze(0)
        diff = x - y
        dis = torch.sum(torch.square(diff), dim=1)
        if torch.min(dis) < 0:
            raise RuntimeError('dis small than 0')
        if square:
            return dis
        else:
            return torch.sqrt(dis)

def knn(x, batch_idx, k:int=16):
    """
    input :     x -> [N, F, 1]
                batch_idx -> [N]
                k -> int
    output :    index -> [N, K]
    """
    with torch.no_grad():
        batch_size = torch.max(batch_idx) + 1
        index_base = torch.zeros([x.shape[0], 1], dtype=torch.long, device=x.device)
        index_list = []
        base = 0
        for bs in range(batch_size):
            x_bs = x[batch_idx==bs]
            dis = pointwise_distance(x_bs.detach(), x_bs.detach())
            _, idx = torch.topk(-dis, k=k)
            index_list.append(idx)
            index_base[batch_idx==bs] = base
            base += len(x_bs)
        index = torch.cat(index_list, dim=0) + index_base
    return index

class PlainGCN(torch.nn.Module):
    def __init__(self, NUM_FILTERS, ATT, MERGE, K, input_channels=4):
        super(PlainGCN, self).__init__()
        self.num_filters = NUM_FILTERS
        self.k = K
        self.act = 'relu'
        self.norm = 'batch'
        self.bias = True
        self.dgn = False
        self.sum = False
        self.fdfs = True
        self.sym = False
        self.merge = MERGE
        self.att = ATT

        if self.sym:
            input_channels = int(input_channels/2)

        channels = [input_channels]
        for f_num in self.num_filters:
            channels += [f_num] if not self.sym else [int(f_num/2)]
        self.channels = channels
        self.num_point_features = channels[-1] if not self.sym else channels[-1]*2
        model_list = []
        for i in range(len(channels) - 1):
            in_c = channels[i]
            out_c = channels[i + 1]
            model_list += [EdgeConv(in_c, out_c, act=self.act, norm=self.norm, bias=self.bias, diss=self.fdfs, merge=self.merge, att=self.att)]

        self.models = ModuleList(model_list)

    def forward(self, pillar_features, voxel_coords):
        coords = voxel_coords.float()
        features = pillar_features.unsqueeze(-1)

        pos = coords[:, 1:4].unsqueeze(-1)
        batch_idx = coords[:, 0].long()

        index = knn(pos, batch_idx, k=self.k)

        for model in self.models:
            if self.dgn:
                index = knn(features, batch_idx, k=self.kz)
            features = model(features, index, pos)

        features = features.squeeze()


        return features, voxel_coords





@VOXEL_ENCODERS.register_module()
class DynamicVFE_GCN(DynamicVFE):

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 NUM_FILTERS=[128, 128, 128],
                 ATT=True,
                 MERGE=True,
                 K=9,
                 return_point_feats=False):
        super(DynamicVFE_GCN, self).__init__(in_channels,
                                             feat_channels,
                                             with_distance,
                                             with_cluster_center,
                                             with_voxel_center,
                                             voxel_size,
                                             point_cloud_range,
                                             norm_cfg,
                                             mode,
                                             fusion_layer,
                                             return_point_feats)
        self.FE = PlainGCN(NUM_FILTERS=NUM_FILTERS, ATT=ATT, MERGE=MERGE, K=K, input_channels=128)


    @force_fp32(out_fp16=True)
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None):
        voxel_feats, voxel_coors = super(DynamicVFE_GCN, self).forward(features, coors, points, img_feats, img_metas)

        gcn_feature, coors = self.FE(voxel_feats, voxel_coors)


        return gcn_feature, coors

    # def init_weights(self, pretrained=None):
    #     pass