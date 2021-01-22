import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.models.backbones_3d.vfe.vfe_template import VFETemplate
from ..gcn.tools import *
from ..gcn.conv import *
from torch.nn import ModuleList


class GCNVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, dataset):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.cylinder = self.model_cfg.CYLINDER
        self.k = self.model_cfg.K
        self.symmetry = self.model_cfg.SYMMETRY

        num_point_features += 0

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)


        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        gcn = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            gcn.append(
                EdgeConv(in_filters, out_filters, act='relu', norm='batch', bias=True, diss=True)
            )
        self.gcns = nn.ModuleList(gcn)

        gcn_s = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            gcn_s.append(
                EdgeConv(in_filters, out_filters, act='relu', norm='batch', bias=True, diss=True)
            )
        self.gcn_s = nn.ModuleList(gcn_s)



    def get_output_feature_dim(self):
        if self.symmetry:
            return self.num_filters[-1]*2
        return self.num_filters[-1]


    def voxels2points(self, voxel_features, voxel_pos, voxel_num_points):
        """
        output: feature -> [N1+N2+..., F]
                pos -> [N1+N2+..., [B_idx, V_idx, x, y, z]]
        """
        voxel_count = voxel_features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        # mask = torch.unsqueeze(mask, -1).type_as(voxel_features)

        voxel_pos = voxel_pos.unsqueeze(1).repeat(1, voxel_count, 1)

        feature = voxel_features.view(-1, voxel_features.shape[-1]).contiguous()
        pos = voxel_pos.view(-1, voxel_pos.shape[-1]).contiguous()
        g_mask = mask.view(-1).contiguous()

        feature = feature[g_mask]
        pos = pos[g_mask]
        return feature, pos, g_mask

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(
            -1, 1, 1)

        feature_mean = voxel_features.sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        feature_mean = feature_mean.repeat([1,voxel_features.shape[1],1])

        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                    coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                    coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                    coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        # useing the voxel pos but not mean of points
        voxel_pos_0 = coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset
        voxel_pos_1 = coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset
        voxel_pos_2 = coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset
        voxel_pos = torch.cat([voxel_pos_0, voxel_pos_1, voxel_pos_2], dim=-1)

        batch_idx = coords[:, 0].unsqueeze(-1)
        voxel_idx = torch.arange(voxel_pos.shape[0], device=voxel_pos.device).unsqueeze(-1)
        voxel_pos = torch.cat([batch_idx, voxel_idx, voxel_pos], dim=-1)
        feature, bidx_vidx_pos, g_mask = self.voxels2points(voxel_features, voxel_pos, voxel_num_points)


        # gcn on entire point cloud
        pos = bidx_vidx_pos[:, -3:].unsqueeze(-1)
        batch_idx = bidx_vidx_pos[:, 0].long()
        feature = feature.unsqueeze(-1)
        if self.symmetry:
            feature_s = feature.clone()
            if 'z' in self.symmetry:
                feature_s[:, 2] *= -1
            if 'r' in self.symmetry:
                feature_s[:, 3] *= -1

        if self.cylinder:
            index = knn(pos[:, :2, :], batch_idx, k=self.k)
        else:
            index = knn(pos, batch_idx, k=self.k)
        for model in self.gcns:
            feature = model(feature, index, pos)
        feature = feature.squeeze(-1)

        if self.symmetry:
            for model in self.gcn_s:
                feature_s = model(feature_s, index, pos)
            feature_s = feature_s.squeeze(-1)
            feature = torch.cat([feature, feature_s], dim=-1)

        voxel_features_new = torch.zeros(voxel_features.shape[0]*voxel_features.shape[1], self.get_output_feature_dim(), device=voxel_features.device)
        voxel_features_new[g_mask] = feature
        voxel_features_new = voxel_features_new.view(voxel_features.shape[0],
                                                     voxel_features.shape[1],
                                                     self.get_output_feature_dim()).contiguous()
        features = torch.max(voxel_features_new, dim=1, keepdim=True)[0].squeeze()

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