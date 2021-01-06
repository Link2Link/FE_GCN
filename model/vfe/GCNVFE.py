# _*_ coding: utf-8 _*_
# @Time : 2021/1/6
# @Author : Chenfei Wang
# @File : GCNVEF.py
# @desc :
# @note :



import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.models.backbones_3d.vfe.vfe_template import VFETemplate
from spconv.utils import VoxelGenerator
import numpy as np

class GCNVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, dataset):
        super().__init__(model_cfg=model_cfg)
        self.num_filters = self.model_cfg.NUM_FILTERS
        self.dataset_cfg = dataset.dataset_cfg
        self.training = dataset.training

        max_num_points = self.dataset_cfg.DATA_PROCESSOR[2].MAX_POINTS_PER_VOXEL
        max_voxels = self.dataset_cfg.DATA_PROCESSOR[2].MAX_NUMBER_OF_VOXELS['train'] if self.training else self.dataset_cfg.DATA_PROCESSOR[2].MAX_NUMBER_OF_VOXELS['test']
        self.voxel_generator = VoxelGenerator(voxel_size=voxel_size,
                                  point_cloud_range=point_cloud_range,
                                  max_num_points=max_num_points,
                                  max_voxels=max_voxels)

    def get_output_feature_dim(self):
        return self.num_filters[-1]


    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        print(voxel_output.keys())
        print(voxel_output['voxels'].shape)
        raise RuntimeError
        pass

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