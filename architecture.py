# _*_ coding: utf-8 _*_
# @Time : 2020/12/18
# @Author : Chenfei Wang
# @File : architecture.py.py
# @desc :
# @note :
from pcdet.models.detectors.detector3d_template import Detector3DTemplate
from model import sampler, backbone, roi_head
from pcdet.models import roi_heads
from model import gcn, vfe

class GCN_Pillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, args):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'GCN',
            'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()

    def build_GCN(self, model_info_dict):
        if self.model_cfg.get('GCN', None) is None:
            return None, model_info_dict
        gcn_module = gcn.__all__[self.model_cfg.GCN.NAME](
            model_cfg=self.model_cfg.GCN,
            input_channels=model_info_dict['num_point_features'],
        )
        model_info_dict['module_list'].append(gcn_module)
        model_info_dict['num_point_features'] = gcn_module.num_point_features
        return gcn_module, model_info_dict

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_point_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            dataset=self.dataset,
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

