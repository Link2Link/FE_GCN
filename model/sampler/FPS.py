# _*_ coding: utf-8 _*_
# @Time : 2020/12/23
# @Author : Chenfei Wang
# @File : FPS.py
# @desc :
# @note :

import torch
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules

class SamplerHead(torch.nn.Module):
    def __init__(self, cfg, num_rawpoint_features, voxel_size, point_cloud_range):
        super(SamplerHead, self).__init__()
        model_cfg = cfg.SAMPLER
        self.NUM_KEYPOINTS = model_cfg.NUM_POINTS
        self.model_cfg = model_cfg
        SA_cfg = self.model_cfg.SA_LAYER
        mlps = SA_cfg['raw_points'].MLPS
        for k in range(len(mlps)):
            mlps[k] = [num_rawpoint_features - 3] + mlps[k]
        c_in = 0
        c_in += sum([x[-1] for x in mlps])
        self.c_in = c_in
        cfg.BACKBONE_3D.c_in = self.c_in
        self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSG(
            radii=SA_cfg['raw_points'].POOL_RADIUS,
            nsamples=SA_cfg['raw_points'].NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method='max_pool'
        )

    def forward(self, batch_dict):
        keypoints = self.get_sampled_points(batch_dict)
        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)
        raw_points = batch_dict['points']
        xyz = raw_points[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()
        point_features = raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None
        point_features_list = []
        pooled_points, pooled_features = self.SA_rawpoints(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features,
        )
        point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))
        point_features = torch.cat(point_features_list, dim=2)
        batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)

        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        batch_dict['point_coords'] = point_coords  # (BxN, 4)
        return batch_dict


    def get_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']
        src_points = batch_dict['points'][:, 1:4]
        batch_indices = batch_dict['points'][:, 0].long()
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                sampled_points[:, :, 0:3].contiguous(), self.NUM_KEYPOINTS
            ).long()

            if sampled_points.shape[1] < self.NUM_KEYPOINTS:
                empty_num = self.NUM_KEYPOINTS - sampled_points.shape[1]
                cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

            keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
            keypoints_list.append(keypoints)
        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints
