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
        channels = [input_channels] + self.num_filters
        self.num_point_features = channels[-1]
        model_list = []
        for i in range(len(channels) - 1):
            in_c = channels[i]
            out_c = channels[i+1]
            model_list += [EdgeConv(in_c, out_c, act='relu', norm='batch', bias=True)]

        self.models = ModuleList(model_list)

    def forward(self, batch_dict):
        # self.print(batch_dict)
        points = batch_dict['points']
        input = points[:, 1:].unsqueeze(-1)
        pos = points[:, 1:4].unsqueeze(-1)

        batch_idx = points[:, 0].long()
        index = knn(pos, batch_idx, k=16)

        feature = input
        for model in self.models:
            feature = model(feature, index)

        batch_dict['gcn_feature'] = feature.squeeze(-1)
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