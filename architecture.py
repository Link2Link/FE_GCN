# _*_ coding: utf-8 _*_
# @Time : 2020/12/18
# @Author : Chenfei Wang
# @File : architecture.py.py
# @desc :
# @note :
from pcdet.models.detectors.detector3d_template import Detector3DTemplate
from model import sampler, backbone, roi_head


class GCNPart2A(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, args):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'sampler', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbone.__all__[self.model_cfg.BACKBONE_3D.NAME](cfg=self.model_cfg)
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        return backbone_3d_module, model_info_dict

    def build_sampler(self, model_info_dict):
        if self.model_cfg.get('SAMPLER', None) is None:
            return None, model_info_dict
        sample_module = sampler.__all__[self.model_cfg.SAMPLER.NAME](
            cfg=self.model_cfg,
            num_rawpoint_features=model_info_dict['num_rawpoint_features'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(sample_module)
        return sample_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_head.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, batch_dict):
        for i, cur_module in enumerate(self.module_list):
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
        loss_point, tb_dict = self.point_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_point + loss_rcnn
        return loss, tb_dict, disp_dict