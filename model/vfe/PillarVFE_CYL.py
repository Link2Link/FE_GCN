# _*_ coding: utf-8 _*_
# @Time : 2021/1/6
# @Author : Chenfei Wang
# @File : vef.py
# @desc :
# @note :

import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.models.backbones_3d.vfe.vfe_template import VFETemplate
from ..gcn.tools import *
from ..gcn.conv import *
from torch.nn import ModuleList

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part])
                               for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, dataset):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.symmetry = self.model_cfg.SYMMETRY
        self.cylinder =  self.model_cfg.CYLINDER

        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)



        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers0 = nn.ModuleList(pfn_layers)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers1 = nn.ModuleList(pfn_layers)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers2 = nn.ModuleList(pfn_layers)



        if self.symmetry:
            pfn_layers_s = []
            for i in range(len(num_filters) - 1):
                in_filters = num_filters[i]
                out_filters = num_filters[i + 1]
                pfn_layers_s.append(
                    PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
                )
            self.pfn_layers_s = nn.ModuleList(pfn_layers_s)


        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1] if not self.symmetry else 2*self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def cylinder_search(self,  voxel_pos, points, voxel_num_points, k=32):
        """
        output : cylinders -> [N, K, 4, cylinders]
                 numbers -> [N, cylinders]
                 masks -> [N, K, 1, cylinders]
        """
        voxel_batch_idx = voxel_pos[:, 0].long()
        points_batch_idx = points[:, 0].long()
        batch_size = torch.max(voxel_batch_idx) + 1

        cylinders = []
        numbers = []
        for bs in range(batch_size):
            point_cur = points[points_batch_idx==bs, 1:]
            voxel_pos_cur = voxel_pos[voxel_batch_idx == bs, 2:]
            num_points_cur = voxel_num_points[voxel_batch_idx==bs]
            dis = pointwise_distance(voxel_pos_cur[:, :2], point_cur[:, :2], square=False)

            cylinder_points = []
            cylinder_numbers = []

            for radius in self.cylinder[::-1]:
                dis[dis > radius * 1.414] = -1
                value, idx = torch.topk(dis, k=k)
                selected_points = index_select(point_cur.unsqueeze(-1), idx).transpose(1,2)
                selected_num = torch.sum(value>-1, dim=-1)
                cylinder_points.append(selected_points)
                cylinder_numbers.append(selected_num)

            cylinder_points = torch.stack(cylinder_points, dim=-1)
            cylinder_numbers = torch.stack(cylinder_numbers, dim=-1)

            cylinders.append(cylinder_points)
            numbers.append(cylinder_numbers)

        cylinders = torch.cat(cylinders, dim=0)
        numbers = torch.cat(numbers, dim=0)

        voxel_count = cylinders.shape[1]
        masks = []
        for i in range(len(self.cylinder)):
            mask = self.get_paddings_indicator(numbers[:, i], voxel_count, axis=0)
            masks.append(mask)
        masks = torch.stack(masks, dim=-1).unsqueeze(2)
        cylinders = cylinders * masks

        return cylinders, numbers, masks


    def forward(self, batch_dict, **kwargs):

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']


        # cylinder
        voxel_pos_0 = coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset
        voxel_pos_1 = coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset
        voxel_pos_2 = coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset
        voxel_pos = torch.cat([voxel_pos_0, voxel_pos_1, voxel_pos_2], dim=-1)

        batch_idx = coords[:, 0].unsqueeze(-1)
        voxel_idx = torch.arange(voxel_pos.shape[0], device=voxel_pos.device).unsqueeze(-1)
        voxel_pos = torch.cat([batch_idx, voxel_idx, voxel_pos], dim=-1)

        points = batch_dict['points']
        cylinders, numbers, masks = self.cylinder_search(voxel_pos, points, voxel_num_points)
        points_mean = cylinders[:, :, :3, :].sum(dim=1, keepdim=True) / numbers.type_as(cylinders).unsqueeze(1).unsqueeze(1)

        f_cluster = cylinders[:, :, :3, :] - points_mean

        f_center = cylinders[:, :, :3, :]- voxel_pos[:, -3:].unsqueeze(1).unsqueeze(-1)



        if self.use_absolute_xyz:
            features = torch.cat([cylinders, f_cluster, f_center], dim=2)
        else:
            features = torch.cat([cylinders[:, :, 3:, :], f_cluster, f_center], dim=2)

        features *= masks

        if self.symmetry:
            s_features = features.clone()
            s_features[:, :, [2, 6, 9], :] *= -1



        for pfn in self.pfn_layers0:
            features0 = pfn(features[..., 0])
        features0 = features0.squeeze()

        for pfn in self.pfn_layers1:
            features1 = pfn(features[..., 1])
        features1 = features1.squeeze()

        for pfn in self.pfn_layers2:
            features2 = pfn(features[..., 2])
        features2 = features2.squeeze()

        features = torch.cat([features0, features1, features2], dim=-1)

        if self.symmetry:
            for pfn in self.pfn_layers_s:
                s_features = pfn(s_features)
            s_features = s_features.squeeze()
            features = torch.cat([features, s_features], dim=1)

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