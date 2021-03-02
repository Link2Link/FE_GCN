import argparse
import glob
from pathlib import Path

# import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from tools.visualization.visualize import visualization
# from visual_utils import visualize_utils as V
from architecture import SATGCN
import sys
import vispy
import copy
# import cv2
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

class visual(visualization):
    def __init__(self, model, model2, demo_dataset):
        super(visual, self).__init__()
        self.model = model
        self.model2 = model2
        self.dataset = demo_dataset
        self.offset = 1000
        self.draw()

    def draw(self):
        with torch.no_grad():
            data_dict = self.dataset[self.offset]

            data_dict = self.dataset.collate_batch([data_dict])
            frame_id = data_dict['frame_id']
            gt_boxes = data_dict['gt_boxes']
            gt_label = gt_boxes[0][:, -1]
            gt_boxes = gt_boxes[0][:, :7]

            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)
            pred_dicts2, _ = self.model2.forward(data_dict)

            channel = [0,1,2]


            img = pred_dicts[0]['spatial_features'].permute(2,3,1,0).squeeze()[:,:, channel]
            img2 = pred_dicts2[0]['spatial_features'].permute(2,3,1,0).squeeze()[:,:, channel]

            img = torch.mean(img, dim=2)
            img2 = torch.mean(img2, dim=2)
            img = img
            img2 = img2

            img = (img - torch.min(img))/(torch.max(img) - torch.min(img))
            img2 = (img2 - torch.min(img2))/(torch.max(img2) - torch.min(img2))


            points = data_dict['points'][:, 1:].cpu().numpy()

            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].long().cpu().numpy()
            img = img.cpu().numpy()
            img2 = img2.cpu().numpy()

            threshold = 0.5 * np.ones_like(pred_scores)
            threshold[pred_labels == 1] = 0.7
            mask = pred_scores > threshold
            pred_boxes = pred_boxes[mask]
            pred_labels = pred_labels[mask]

            pred_boxes2 = pred_dicts2[0]['pred_boxes'].cpu().numpy()
            pred_scores2 = pred_dicts2[0]['pred_scores'].cpu().numpy()
            pred_labels2 = pred_dicts2[0]['pred_labels'].cpu().numpy()

            threshold = 0.5 * np.ones_like(pred_labels2)
            threshold[pred_labels2 == 1] = 0.7
            mask = pred_scores2 > threshold
            pred_boxes2 = pred_boxes2[mask]
            pred_labels2 = pred_labels2[mask]


            print(f'Visualized sample index: \t{self.offset}', 'frame id:', frame_id)


        # print(pred_scores)
        points = points[:, :3]

        D2 = False
        # self.draw_bbox(pred_boxes, pred_labels, flag=0, D2=D2)
        self.draw_points(points[:, :3], flag=0, D2=D2)
        self.draw_img(img, flag=1)
        self.draw_img(img2, flag=2)

        # self.draw_bbox(pred_boxes2, pred_labels2, flag=1, D2=D2)
        # self.draw_points(points[:, :3], flag=1, D2=D2)
        #
        # self.draw_bbox(gt_boxes, gt_label.astype(int), flag=2, D2=D2)
        # self.draw_points(points[:, :3], flag=2, D2=D2)

    # self.draw_text(text=pred_scores, label=pred_labels, font_size=1, boxes=pred_boxes)

    def press_N(self):
        self.offset += 1 if self.offset < len(self.dataset)-1 else 0
        self.draw()

    def press_B(self):
        self.offset -= 1 if self.offset > 0 else 0
        self.draw()

    # def press_butten(self, key):
    #     if key == 'W':
    #         pass



class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/FE_DISS_ATM.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/home/llx/dataset/kitti',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str,
                        default='/home/llx/work_dir/output/pointpillar/default/ckpt/checkpoint_pointpillar_baseline.pth',
                        help='specify the pretrained model')
    parser.add_argument('--ckpt2', type=str,
                        default='/home/llx/work_dir/output/checkpoint_FE_FDFS_ATM_best.pth',
                        help='specify the pretrained model')

    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg2 = copy.deepcopy(cfg)
    del cfg.MODEL['GCN']
    return args, cfg, cfg2



def main():
    args, cfg, cfg2 = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    # model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model = SATGCN(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset, args=args)
    model2 = SATGCN(model_cfg=cfg2.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset, args=args)

    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model2.load_params_from_file(filename=args.ckpt2, logger=logger, to_cpu=True)

    model.cuda()
    model.eval()
    model2.cuda()
    model2.eval()



    V = visual(model, model2, demo_dataset)
    if sys.flags.interactive != 1:
        vispy.app.run()


if __name__ == '__main__':
    main()
