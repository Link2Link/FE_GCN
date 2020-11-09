import __init__
import numpy as np
import torch
from torch_geometric.data import DenseDataLoader
# import architecture
import utils.panda_dataset as dataset
import logging
from tqdm import tqdm
import os, sys
# import config
from DetecterData import *
import config
import vispy
from vispy.scene import visuals, SceneCanvas
from utils import color, visualize, graph, tools
from utils.tools import *

from sem_pretrain.code import architecture as semseg
from sem_pretrain.code import config
from gcn_lib.dense import BasicConv, GraphConv2d, PlainDynBlock2d, ResDynBlock2d, ResBlock2d, MLP
from torch.nn import Sequential as Seq
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
import torch.nn as nn
from torch.nn import Sequential as Seq
from itertools import chain
import math

class Nomalizer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Nomalizer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.T_matrix = Parameter(torch.Tensor(out_channels-3, in_channels))

        self.reset_parameters()

    def reset_parameters(self) -> None:

        kaiming_uniform_(self.T_matrix, mode='fan_in', nonlinearity='relu')


    def forward(self, vertex):
        position = torch.mean(vertex, dim=0)
        vertex = vertex - position
        vertex = vertex.transpose(0,1) # transpose dim
        vertex_feature = torch.mm(self.T_matrix, vertex)
        feature = torch.cat([vertex, vertex_feature], dim=0)
        feature = torch.max(feature, dim = 1)[0].unsqueeze(0)
        position = position.unsqueeze(0)
        return position, feature


class ObjectDetection(torch.nn.Module):
    def __init__(self, channels, dropout=0.3):
        super(ObjectDetection, self).__init__()

        m = []
        for i in range(1, len(channels)-1):
            m.append(nn.Linear(channels[i - 1], channels[i], True))
            m.append(nn.ReLU())
            m.append(nn.BatchNorm1d(channels[i]))
        m.append(nn.Dropout(p=dropout),)
        m.append(nn.Linear(channels[-2], channels[-1]))
        self.MLP = Seq(*m)

    def forward(self, inputs):
        return self.MLP(inputs)


class visual(visualize.visualization):
    global cube_idx
    global v_list
    global points
    def __init__(self):
        super(visual, self).__init__()
        self.offset = 0

    def press_N(self):
        global points
        self.offset += 1
        if self.offset > len(cube_idx):
            self.offset = len(cube_idx)
        self.draw_bbox(cube_idx[self.offset, :], flag=1)
        print(self.offset)




    def press_B(self):
        global points
        self.offset -= 1
        if self.offset < 0:
            self.offset = 0
        self.draw_bbox(cube_idx[self.offset, :], flag=1)
        print(self.offset)

if __name__ == '__main__':
    import random, numpy as np, argparse

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = 2
    device = 'cuda'

    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--in_channels', default=3, type=int, help='input channels (default:9)')
    parser.add_argument('--n_classes', default=2, type=int, help='num of segmentation classes (default:13)')
    parser.add_argument('--block', default='res', type=str, help='graph backbone block type {plain, res, dense}')
    parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
    parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
    parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
    parser.add_argument('--bias', default=True, type=bool, help='bias of conv layer True or False')
    parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
    parser.add_argument('--n_blocks', default=7, type=int, help='number of basic blocks')
    parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')
    args = parser.parse_args()

    dataset_list = [DetectionDataset(seq_num=i) for i in range(40)]
    dataset = DatasetConcat(dataset_list)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    nomallized_feature = 256
    nomalizer = Nomalizer(in_channels=3, out_channels=nomallized_feature).to(device)
    detector = ObjectDetection(channels=[nomallized_feature, 256, 64, 2]).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(chain(nomalizer.parameters(), detector.parameters()), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_adjust_freq, opt.lr_decay_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_adjust_freq, opt.lr_decay_rate)

    for index in loader:
        data_list = dataset.data[index]
        obj_points_list, label, boxes = DatasetConcat.UnpackData(data_list, device)

        optimizer.zero_grad()

        # nomalize
        features = torch.empty([0, nomallized_feature]).to(device)
        positions = torch.empty([0, 3]).to(device)
        for vertex in obj_points_list:
            position, feature = nomalizer(vertex)
            features = torch.cat([features, feature], dim=0)
            positions = torch.cat([positions, position], dim=0)

        # classify
        outputs = detector(features)
        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()
        break

    # #
    # v = visual()
    # v.draw_semseg(points, pred, flag=0)
    # v.draw_semseg(points[vertex], pred[vertex], flag=1)
    # #
    # v.draw_bbox(cube_idx, flag=0)
    # # v.draw_bbox(cube_idx, flag=1)
    # if sys.flags.interactive != 1:
    #     vispy.app.run() #开始运行
