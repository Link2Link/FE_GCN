# _*_ coding: utf-8 _*_
# @Time : 2020/12/23
# @Author : Chenfei Wang
# @File : VFEC.py
# @desc :
# @note :

import torch
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from pcdet.utils import common_utils
import numpy as np

class VFEC(torch.nn.Module):
    def __init__(self, cfg, num_rawpoint_features, voxel_size, point_cloud_range):
        super(VFEC, self).__init__()
        model_cfg = cfg.SAMPLER
        self.NUM_KEYPOINTS = model_cfg.NUM_POINTS
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.c_in = 4
        cfg.BACKBONE_3D.c_in = self.c_in

    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        point_coords = common_utils.get_voxel_centers(
            voxel_coords[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )

        point_coords = torch.cat((voxel_coords[:, 0:1], point_coords), dim=1)
        batch_size = batch_dict['batch_size']
        point_coord_list = []
        voxel_feature_list = []
        for bs_idx in range(batch_size):
            point = point_coords[point_coords[:, 0].long() == bs_idx]
            feature = voxel_features[point_coords[:, 0].long() == bs_idx]
            num = len(point)
            if num < self.NUM_KEYPOINTS:
                index = np.arange(num)
                np.random.shuffle(index)
                index = index[:self.NUM_KEYPOINTS-num]
                index = np.append(np.arange(num), index)
                point = point[index]
                feature = feature[index]
            point_coord_list.append(point)
            voxel_feature_list.append(feature)
        point_coords = torch.cat(point_coord_list, dim=0)
        voxel_features = torch.cat(voxel_feature_list, dim=0)
        batch_dict['point_features'] = voxel_features
        batch_dict['point_coords'] = point_coords  # (BxN, 4)
        return batch_dict


class VFECSA(torch.nn.Module):
    def __init__(self, cfg, num_rawpoint_features, voxel_size, point_cloud_range):
        super(VFECSA, self).__init__()
        model_cfg = cfg.SAMPLER
        self.NUM_KEYPOINTS = model_cfg.NUM_POINTS
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        SA_cfg = self.model_cfg.SA_LAYER
        mlps = SA_cfg['raw_points'].MLPS
        for k in range(len(mlps)):
            mlps[k] = [num_rawpoint_features - 3] + mlps[k]
        c_in = 0
        c_in += sum([x[-1] for x in mlps])
        self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSG(
            radii=SA_cfg['raw_points'].POOL_RADIUS,
            nsamples=SA_cfg['raw_points'].NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method='max_pool'
        )
        self.c_in = c_in + 4
        cfg.BACKBONE_3D.c_in = self.c_in

    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        point_coords = common_utils.get_voxel_centers(
            voxel_coords[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )

        point_coords = torch.cat((voxel_coords[:, 0:1], point_coords), dim=1)
        batch_size = batch_dict['batch_size']
        point_coord_list = []
        voxel_feature_list = []
        for bs_idx in range(batch_size):
            point = point_coords[point_coords[:, 0].long() == bs_idx]
            feature = voxel_features[point_coords[:, 0].long() == bs_idx]
            num = len(point)
            if num < self.NUM_KEYPOINTS:
                index = np.arange(num)
                np.random.shuffle(index)
                index = index[:self.NUM_KEYPOINTS-num]
                index = np.append(np.arange(num), index)
                point = point[index]
                feature = feature[index]
            point_coord_list.append(point)
            voxel_feature_list.append(feature)
        point_coords = torch.cat(point_coord_list, dim=0)
        voxel_features = torch.cat(voxel_feature_list, dim=0)

        new_xyz = point_coords[:, 1:].contiguous()
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(self.NUM_KEYPOINTS)
        raw_points = batch_dict['points']
        xyz = raw_points[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()
        point_features = raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None
        pooled_points, pooled_features = self.SA_rawpoints(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features,
        )
        point_features = torch.cat([voxel_features, pooled_features], dim=1)
        batch_dict['point_features'] = point_features
        batch_dict['point_coords'] = point_coords  # (BxN, 4)
        return batch_dict
