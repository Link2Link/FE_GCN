import torch
from torch.nn import Sequential as Seq
from torch.nn import ModuleList
from .tools import *
from .conv import EdgeConv
data_collect = []




class ResGCN(torch.nn.Module):
    def __init__(self, model_cfg, input_channels=4):
        super(ResGCN, self).__init__()
        self.num_filters = model_cfg.NUM_FILTERS
        self.act = model_cfg.ACT
        self.norm = model_cfg.NORM
        self.bias = model_cfg.BIAS
        self.dgn = model_cfg.DYN_GRAPH
        self.sum = model_cfg.SUM
        self.relu = model_cfg.RELU
        self.diss = model_cfg.DISS


        channels = [input_channels] + self.num_filters
        self.num_point_features = channels[-1]
        model_list = []
        for i in range(len(channels) - 1):
            in_c = channels[i]
            out_c = channels[i + 1]
            model_list += [EdgeConv(in_c, out_c, act=self.act, norm=self.norm, bias=self.bias, diss=self.diss)]

        self.models = ModuleList(model_list)

    def forward(self, batch_dict):
        coords = batch_dict['voxel_coords']
        features = batch_dict['pillar_features'].unsqueeze(-1)

        pos = coords[:, 1:4].unsqueeze(-1)
        batch_idx = coords[:, 0].long()

        if not self.dgn:  # static graph
            index = knn(pos, batch_idx, k=16)
            for model in self.models:
                features = torch.relu(features + model(features, index, pos)) if self.relu \
                    else features + model(features, index, pos)
        else:
            for model in self.models:
                index = knn(features, batch_idx, k=16)
                features = torch.relu(features + model(features, index, pos)) if self.relu \
                    else features + model(features, index, pos)

        features = features.squeeze()

        if self.sum:
            batch_dict['pillar_features'] = batch_dict['pillar_features'] + features
        else:
            batch_dict['pillar_features'] = features
        return batch_dict

    def print(self, batch_dict):
        print('++++++++++++++++++++++++++++++++++++++++')
        print(batch_dict.keys())
        for k, v in batch_dict.items():
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