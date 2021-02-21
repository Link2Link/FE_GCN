

import torch
from torch.nn import Sequential as Seq
from torch.nn import ModuleList
from .tools import *
from .conv import EdgeConv

class PlainGCN(torch.nn.Module):
    def __init__(self, model_cfg, input_channels=4):
        super(PlainGCN, self).__init__()
        self.num_filters = model_cfg.NUM_FILTERS
        self.k = model_cfg.K
        self.act = model_cfg.ACT
        self.norm = model_cfg.NORM
        self.bias = model_cfg.BIAS
        self.dgn = model_cfg.DYN_GRAPH
        self.sum = model_cfg.SUM
        self.diss = model_cfg.DISS
        self.sym = model_cfg.SYMMETRY
        self.merge = model_cfg.MERGE
        self.att = model_cfg.ATT

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
            model_list += [EdgeConv(in_c, out_c, act=self.act, norm=self.norm, bias=self.bias, diss=self.diss, merge=self.merge, att=self.att)]

        self.models = ModuleList(model_list)

        if self.sym:
            model_s_list = []
            for i in range(len(channels) - 1):
                in_c = channels[i]
                out_c = channels[i + 1]
                model_s_list += [EdgeConv(in_c, out_c, act=self.act, norm=self.norm, bias=self.bias, diss=self.diss)]
            self.models_s = ModuleList(model_s_list)

    def forward(self, batch_dict):
        coords = batch_dict['voxel_coords']
        features = batch_dict['pillar_features'].unsqueeze(-1)
        if self.sym:
            feature_o = features[:, :self.channels[0], :]
            feature_s = features[:, self.channels[0]:, :]

        pos = coords[:, 1:4].unsqueeze(-1)
        batch_idx = coords[:, 0].long()

        index = knn(pos, batch_idx, k=self.k)
        if self.sym:
            for model, model_s in zip(self.models, self.models_s):
                if self.dgn:
                    index = knn(feature_o, batch_idx, k=9)
                feature_o = model(feature_o, index, pos)
                feature_s = model_s(feature_s, index, pos)
            features = torch.cat([feature_o, feature_s], dim=1)
        else:
            for model in self.models:
                if self.dgn:
                    index = knn(features, batch_idx, k=9)
                features = model(features, index, pos)

        features = features.squeeze()

        if self.sum:
            batch_dict['pillar_features'] = batch_dict['pillar_features'] + features
        else:
            batch_dict['pillar_features'] = features
        return batch_dict

    def print(self, batch_dict):
        print('++++++++++++++++++++++++++++++++++++++++')
        print(batch_dict.keys())
        for k,v in batch_dict.items():
            try:
                shape = v.shape
                if len(shape) <= 2:
                    print(k, v.shape)
                    print(v)
                else:
                    print(k, v.shape)

            except:
                print(k, v)
        print('----------------------------------------')

class PlainGCN3D(torch.nn.Module):
    def __init__(self, model_cfg, input_channels=4):
        super(PlainGCN3D, self).__init__()
        self.num_filters = model_cfg.NUM_FILTERS
        self.k = model_cfg.K
        self.act = model_cfg.ACT
        self.norm = model_cfg.NORM
        self.bias = model_cfg.BIAS
        self.dgn = model_cfg.DYN_GRAPH
        self.sum = model_cfg.SUM
        self.diss = model_cfg.DISS
        self.sym = model_cfg.SYMMETRY
        self.merge = model_cfg.MERGE

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
            model_list += [EdgeConv(in_c, out_c, act=self.act, norm=self.norm, bias=self.bias, diss=self.diss, merge=self.merge)]

        self.models = ModuleList(model_list)

        if self.sym:
            model_s_list = []
            for i in range(len(channels) - 1):
                in_c = channels[i]
                out_c = channels[i + 1]
                model_s_list += [EdgeConv(in_c, out_c, act=self.act, norm=self.norm, bias=self.bias, diss=self.diss)]
            self.models_s = ModuleList(model_s_list)

    def forward(self, batch_dict):
        coords = batch_dict['voxel_coords']
        features = batch_dict['voxel_features'].unsqueeze(-1)
        pos = coords[:, 1:4].unsqueeze(-1)
        batch_idx = coords[:, 0].long()

        index = knn(pos, batch_idx, k=self.k)
        if self.sym:
            for model, model_s in zip(self.models, self.models_s):
                if self.dgn:
                    index = knn(feature_o, batch_idx, k=9)
                feature_o = model(feature_o, index, pos)
                feature_s = model_s(feature_s, index, pos)
            features = torch.cat([feature_o, feature_s], dim=1)
        else:
            for model in self.models:
                if self.dgn:
                    index = knn(features, batch_idx, k=9)
                features = model(features, index, pos)

        features = features.squeeze()

        if self.sum:
            batch_dict['voxel_features'] = batch_dict['voxel_features'] + features
        else:
            batch_dict['voxel_features'] = features
        return batch_dict

    def print(self, batch_dict):
        print('++++++++++++++++++++++++++++++++++++++++')
        print(batch_dict.keys())
        for k,v in batch_dict.items():
            try:
                shape = v.shape
                if len(shape) <= 2:
                    print(k, v.shape)
                    print(v)
                else:
                    print(k, v.shape)

            except:
                print(k, v)
        print('----------------------------------------')