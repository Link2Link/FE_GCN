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
from torch.utils.data import ConcatDataset
import vispy
from vispy.scene import visuals, SceneCanvas
from utils import color, visualize, graph, tools
from torch.utils.data import Dataset, DataLoader

from utils.tools import *
import os
import argparse
import shutil
import random
import numpy as np
import torch
import logging
import logging.config
import pathlib
import glob
import time
import sys
from sem_pretrain.code import architecture as semseg
from sem_pretrain.code import config

class Data_stucture:
    def __init__(self, points):
        self.points = points
        self.index_list = []
        self.flag = np.empty(0, dtype=bool)
        self.cube = np.empty([0, 7], dtype=float)

    def add_obj(self, index, flag, cube):
        self.index_list.append(index)
        self.flag = np.append(self.flag, flag)
        self.cube = np.append(self.cube, np.array([cube]), axis=0)

    def __len__(self):
        return len(self.flag)

    @property
    def num_of_box(self):
        return len(self.flag_index)

    @property
    def flag_index(self):
        index = np.arange(len(self))
        return index[self.flag]

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

class OptInit:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Make data for detection')

        # base
        parser.add_argument('--root_dir', type=str, default='/home/llx/dataset/pandaset/detection', help='the dir for detection data')
        parser.add_argument('--order', type=int, default=0, help='the seq of dataset')

        # dataset args
        parser.add_argument('--data_dir', type=str, default='/home/llx/dataset/pandaset')
        parser.add_argument('--in_channels', default=4, type=int, help='the channel size of input point cloud ')

        # model args
        parser.add_argument('--pretrained_model', type=str, help='path to pretrained model(default: none)',
                            default="/home/llx/code/GCN_Detecter/example/detection/sem_pretrain/deepgcn_ckpt_30_best.pth")
        parser.add_argument('--block', default='res', type=str, help='graph backbone block type {plain, res, dense}')
        parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
        parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
        parser.add_argument('--bias', default=True,  type=bool, help='bias of conv layer True or False')
        parser.add_argument('--n_filters', default=8, type=int, help='number of channels of deep features')
        parser.add_argument('--n_blocks', default=7, type=int, help='number of basic blocks')
        parser.add_argument('--dropout', default=0.3, type=float, help='ratio of dropout')

        args = parser.parse_args()

        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self._set_seed()

    def get_args(self):
        return self.args

    def _set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DetectionDataset(Dataset):
    def __init__(self, seq_num=0 ,dir='/home/llx/dataset/pandaset/detection'):
        super(DetectionDataset, self).__init__()
        self.dir = dir
        self.seq_num = seq_num
        _, _, files = next(os.walk(dir))
        files.sort()
        self.file = files[0]
        self.data = np.load(os.path.join(dir, self.file), allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return index

    @staticmethod
    def append_data(dataset_list):
        L = np.empty(0, dtype=np.object)
        for dataset in dataset_list:
            L = np.append(L, dataset.data)
        return L

class DatasetConcat(ConcatDataset):
    def __init__(self, datasets):
        super(DatasetConcat, self).__init__(datasets=datasets)
        self.data = DetectionDataset.append_data(datasets)

    @staticmethod
    def UnpackData(data_list, device='cuda'):
        obj_points_list = []
        label = torch.tensor([], dtype=torch.long).to(device)
        boxes = torch.tensor([], dtype=torch.float).to(device)

        for data in data_list:
            points = torch.Tensor(data.points).to(device)
            for vertex in data.index_list:
                obj_points_list.append(points[vertex])
            label = torch.cat([label, torch.tensor(data.flag).long().to(device)])
            boxes = torch.cat([boxes, torch.Tensor(data.cube).to(device)], axis=0)

        return obj_points_list, label, boxes









# if __name__ == '__main__':
#     opt = OptInit().get_args()
#     opt.n_classes = 5
#     seq_num = dataset.return_seq(opt.data_dir)
#     seq = seq_num[opt.order]
#     if opt.order >= len(seq_num):
#         raise RuntimeError("order is larger than " + len(seq_num))
#     print("processing ", seq)
#     dataset = dataset.PANDASET(root=opt.data_dir, seq_num=seq)
#     test_loader = DenseDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
#     loader = test_loader
#
#     model = semseg.DenseDeepGCN(opt).to(opt.device)
#     checkpoint = torch.load(opt.pretrained_model)
#     ckpt_model_state_dict = checkpoint['state_dict']
#     model.load_state_dict(ckpt_model_state_dict)
#     model.requires_grad_(False)
#
#
#     data_list = []
#     for i, data in tqdm(enumerate(loader)):
#         data = data.to(opt.device)
#         inputs = torch.cat((data.pos.transpose(1, 2).unsqueeze(3), data.x.unsqueeze(1).unsqueeze(3)), 1)
#         gt = data.y.to(opt.device).long()
#         out = model(inputs, data.edge_index)
#         pred = out.max(dim=1)[1]
#
#         points = data.pos[0].cpu().numpy()
#         pred = pred[0].cpu().numpy()
#         label = gt[0].cpu().numpy()
#         edge_index = data.edge_index[0].cpu().numpy()
#         G = graph.G(len(label), edge_index, label)
#         G_p = graph.G(len(label), edge_index, pred)
#         car_pose = data.car_pose[0].cpu().numpy()
#
#
#         seq = '%03d' % (data.idx[0][0].cpu().numpy())
#         idx = data.idx[0][1].cpu().numpy()
#         cube = np.load(os.path.join(dataset.raw_dir, 'lidar_cuboids_%0s.npy'%seq))
#         cube_idx = cube[cube[:, 0] == idx, 1:]
#         cube_idx[:, :3] -= car_pose
#
#         v_list = G_p.graph_cut(label=1)
#
#         data_stur = Data_stucture(points)
#         for vertex in v_list:
#             if len(vertex) < 20:
#                 continue
#             flag = False
#             for cube in cube_idx:
#                 if tools.box_check(points[vertex], cube):
#                     flag = True
#                     data_stur.add_obj(vertex, flag, cube)
#                     # print("entire {0} objs, new obj {1} points".format(len(data_stur), len(data_stur.index_list[-1])))
#             if flag is not True:
#                 data_stur.add_obj(vertex, flag, np.ones(7)*np.inf)
#
#         data_list.append(data_stur)
#     file_name = os.path.join(opt.root_dir, 'detection_%s.npy'%seq)
#     np.save(file_name, data_list)
#     print("saving file to ", file_name)

if __name__ == '__main__':
    dataset_list = [DetectionDataset(seq_num=i) for i in range(40)]
    dataset = DatasetConcat(dataset_list)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    for index in loader:
        break

    #
    # v = visual()
    # v.draw_semseg(points, pred, flag=0)
    #
    # p = np.empty(0, dtype=int)
    # for index in [pair[0] for pair in cube_point_pair_list]:
    #     p = np.append(p, index)
    # v.draw_semseg(points[p], pred[p], flag=1)
    # #
    # v.draw_bbox(cube_idx, flag=0)
    #
    # b = np.empty([0, 7])
    # for pair in cube_point_pair_list:
    #     if pair[1]:
    #         cube = pair[2]
    #         cube = np.array([cube])
    #         b = np.append(b, cube, axis=0)
    #
    # v.draw_bbox(b, flag=1)
    # if sys.flags.interactive != 1:
    #     vispy.app.run() #开始运行
