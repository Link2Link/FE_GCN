import numpy as np
import spconv
import torch
import torch.nn as nn

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.models.roi_heads import RoIHeadTemplate
from ..backbone.blocks.gcn_basic import *
from ..backbone.blocks.data import *
from ..backbone.blocks.gcn import *



class gcn_roi(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.gcn_point_num = self.model_cfg.GCN_POINT_NUM

        self.roiaware_pool3d_layer = roiaware_pool3d_utils.RoIAwarePool3d(
            out_size=self.model_cfg.ROI_AWARE_POOL.POOL_SIZE,
            max_pts_each_voxel=self.model_cfg.ROI_AWARE_POOL.MAX_POINTS_PER_VOXEL
        )
        c0 = self.model_cfg.ROI_AWARE_POOL.NUM_FEATURES // 2
        self.point_featureConv1 = GraphConv(16, 64, DISS=True)
        self.point_featureConv2 = GraphConv(64, c0, DISS=True)

        self.part_featureConv1 = GraphConv(4, 64, DISS=True)
        self.part_featureConv2 = GraphConv(64, c0, DISS=True)

        self.conv = ResGCN(c0*2, 6, DISS=True)

        self.cls_layers = self.make_fc_layers(
            input_channels=32768, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=32768,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )



    # self.init_weights(weight_init='xavier')


    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def print(self, dict, name='dict'):
        print('++++++++++++++++++++++'+name+'++++++++++++++++++++++')
        for k,v in dict.items():
            try:
                print(k, v.shape)
            except:
                print(k, v)
    def roiaware_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        batch_idx = batch_dict['point_coords'][:, 0]
        point_coords = batch_dict['point_coords'][:, 1:4]
        point_features = batch_dict['point_features']
        part_features = torch.cat((
            batch_dict['point_part_offset'] if not self.model_cfg.get('DISABLE_PART', False) else point_coords,
            batch_dict['point_cls_scores'].view(-1, 1).detach()
        ), dim=1)
        part_features[part_features[:, -1] < self.model_cfg.SEG_MASK_SCORE_THRESH, 0:3] = 0


        rois = batch_dict['rois']

        pooled_part_features_list, pooled_rpn_features_list = [], []
        pooled_coords_list = []

        for bs_idx in range(batch_size):
            bs_mask = (batch_idx == bs_idx)
            cur_point_coords = point_coords[bs_mask]
            cur_part_features = part_features[bs_mask]
            cur_rpn_features = point_features[bs_mask]
            cur_roi = rois[bs_idx][:, 0:7].contiguous()  # (N, 7)

            pooled_part_features = self.roiaware_pool3d_layer.forward(
                cur_roi, cur_point_coords, cur_part_features, pool_method='avg'
            )  # (N, out_x, out_y, out_z, 4)

            pooled_rpn_features = self.roiaware_pool3d_layer.forward(
                cur_roi, cur_point_coords, cur_rpn_features, pool_method='max'
            )  # (N, out_x, out_y, out_z, C)

            pooled_coords = self.roiaware_pool3d_layer.forward(
                cur_roi, cur_point_coords, cur_point_coords, pool_method='avg'
            )  # (N, out_x, out_y, out_z, 3)

            pooled_part_features_list.append(pooled_part_features)
            pooled_rpn_features_list.append(pooled_rpn_features)
            pooled_coords_list.append(pooled_coords)

        pooled_part_features = torch.cat(pooled_part_features_list, dim=0)  # (B * N, out_x, out_y, out_z, 4)
        pooled_rpn_features = torch.cat(pooled_rpn_features_list, dim=0)  # (B * N, out_x, out_y, out_z, C)
        pooled_coords = torch.cat(pooled_coords_list, dim=0)

        return pooled_part_features, pooled_rpn_features, pooled_coords


    def forward(self, batch_dict):
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_part_features, pooled_rpn_features, pooled_coords = self.roiaware_pool(batch_dict)
        batch_size_rcnn = pooled_part_features.shape[0]

        # transform to graph
        num_points = pooled_part_features.shape[1] * pooled_part_features.shape[2] * pooled_part_features.shape[3]

        pooled_part_features = pooled_part_features.view(batch_size_rcnn, num_points, -1)
        pooled_rpn_features = pooled_rpn_features.view(batch_size_rcnn, num_points, -1)
        pooled_coords = pooled_coords.view(batch_size_rcnn, num_points, -1)

        gcn_index = torch.sum(torch.abs(pooled_coords), dim=-1)
        _, index = torch.topk(gcn_index, self.gcn_point_num, dim=-1)

        gcn_part_feature_list = []
        gcn_point_feature_list = []
        pos_list = []
        pt_inside_list = []
        for i in range(len(index)):
            pt_inside_list.append(torch.sum(gcn_index[i, index[i]] > 0))
            pos_list.append(pooled_coords[i, index[i], :])
            gcn_part_feature_list.append(pooled_part_features[i, index[i], :])
            gcn_point_feature_list.append(pooled_rpn_features[i, index[i], :])

        pos = torch.stack(pos_list).unsqueeze(-1).transpose(1,2)
        gcn_part_feature = torch.stack(gcn_part_feature_list).unsqueeze(-1).transpose(1,2)
        gcn_point_feature = torch.stack(gcn_point_feature_list).unsqueeze(-1).transpose(1,2)
        pt_inside = torch.stack(pt_inside_list)

        point_featre_g = Graph(pos, gcn_point_feature, k=20)
        part_featre_g = Graph(pos, gcn_part_feature, k=20)


        point_featre_g = self.point_featureConv2.forward2(self.point_featureConv1.forward2(point_featre_g))
        part_featre_g = self.part_featureConv2.forward2(self.part_featureConv1.forward2(part_featre_g))

        feature = torch.cat([point_featre_g.x, part_featre_g.x], dim=1)
        graph = Graph(pos, feature, k=20)


        graph = self.conv(graph)

        feature = graph.x.view(batch_size_rcnn, -1).unsqueeze(-1)

        rcnn_cls = self.cls_layers(feature)
        rcnn_reg = self.reg_layers(feature)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict