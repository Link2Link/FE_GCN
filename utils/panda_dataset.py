from pandaset import DataSet
import pandas as pd
import sys, os
import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
from sklearn.neighbors import BallTree
from tqdm import tqdm
import gc
import copy
import torch
SamplingNum = 4096*20
            # 0-other 1-car 2-people 3-road 4-Vegetation  41 labels map to 5 labels
            #0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2
label_map = [0,0,0,0,0,4,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,0,0,0,0,0,0,0,0,0,0,0]
label_map = np.array(label_map)

def return_seq(root="/home/llx/dataset/pandaset"):
    dataset = DataSet(os.path.join(root, 'original'))
    seq = dataset.sequences(with_semseg=True)
    seq.sort()
    return seq

class PANDASET(InMemoryDataset):
    def __init__(self,
                 root="/home/llx/dataset/pandaset",
                 seq_num='001',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.root = root
        self.dataset = DataSet(os.path.join(root, 'original'))
        self.seq = self.dataset.sequences(with_semseg=True)
        self.seq.sort()
        self.seq_num = seq_num
        if self.seq_num not in self.seq:
            raise RuntimeError("seq_num not in seq")
        super(PANDASET, self).__init__(root, transform, pre_transform, pre_filter)
        (self.data, self.slices) = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["lidar_data_%s.npy"%(self.seq_num), "lidar_poses_%s.npy"%(self.seq_num), "lidar_semseg_%s.npy"%(self.seq_num), "lidar_cuboids_%s.npy"%(self.seq_num)]

    @property
    def processed_file_names(self):
        return ['%s.pt'%(self.seq_num)]


    def download(self):
        # load data from original to raw
        dataset = DataSet(os.path.join(self.root, 'original'))
        seq_data = dataset[self.seq_num]
        seq_data.load_lidar().load_cuboids().load_semseg()
        seq_data.lidar.set_sensor(0)
        pc_all = seq_data.lidar[:]
        semseg_all = seq_data.semseg[:]
        cuboid_all = seq_data.cuboids[:]

        lidar_data = np.zeros([len(pc_all), SamplingNum, 4])
        lidar_poses = np.zeros([len(pc_all), 3])
        lidar_semseg = np.zeros([len(pc_all), SamplingNum], dtype=np.int)
        for i in range(len(cuboid_all)):
            cuboid_all[i] = cuboid_all[i][['label', 'position.x', 'position.y', 'position.z',
                                           'dimensions.x', 'dimensions.y', 'dimensions.z','yaw']]

        lidar_cuboids_car = np.empty([0, 8])
        for num in range(len(cuboid_all)):
            cuboid_car = cuboid_all[num][cuboid_all[num]['label'] == 'Car']
            bbox = cuboid_car[['position.x', 'position.y', 'position.z',
                               'dimensions.x', 'dimensions.y', 'dimensions.z','yaw']].values
            num_id = np.ones([len(bbox), 1])*num
            bbox = np.concatenate((num_id, bbox), axis=1)
            lidar_cuboids_car = np.concatenate((lidar_cuboids_car, bbox), axis=0)


        lidar_cuboids = lidar_cuboids_car   # only use car bbox

        for idx, (single_scan, single_pose, single_semseg) in enumerate(zip(pc_all, seq_data.lidar.poses, semseg_all)):
            # set pose
            lidar_poses[idx, :] = np.array(list(single_pose['position'].values()))

            # sampling based on index
            index = np.arange(len(single_scan))
            np.random.shuffle(index)
            index = index[:SamplingNum]

            # pick data by index
            sampled_scan = single_scan.loc[index]
            lidar_data[idx, :, :] = sampled_scan.iloc[:, [0, 1, 2, 3]].values

            # pick semantic label by index
            sampled_semseg = single_semseg.loc[index]
            label = sampled_semseg.values.flatten()
            lidar_semseg[idx, :] = label_map[label]
            lidar_semseg = lidar_semseg.astype(np.int)

        np.save(os.path.join(self.raw_dir, self.raw_file_names[0]), lidar_data)
        np.save(os.path.join(self.raw_dir, self.raw_file_names[1]), lidar_poses)
        np.save(os.path.join(self.raw_dir, self.raw_file_names[2]), lidar_semseg)
        np.save(os.path.join(self.raw_dir, self.raw_file_names[3]), lidar_cuboids)


        # # clean memory
        # del dataset, seq_data, lidar_data, lidar_poses, lidar_semseg, lidar_cuboids
        # gc.collect()

    def process(self):

        # process lidar_data
        data_list = []
        lidar_data = np.load(self.raw_paths[0])
        lidar_pos = np.load(self.raw_paths[1])
        lidar_semseg = np.load(self.raw_paths[2])
        lidar_cuboids = np.load(self.raw_paths[3])

        for k in tqdm(range(len(lidar_data))):
            s_data = lidar_data[k, :, :]
            s_pose = lidar_pos[k, :]
            s_semseg = lidar_semseg[k, :]
            pos = s_data[:, :3] - s_pose
            tree = BallTree(pos, leaf_size=8)
            dist, ind = tree.query(pos, k=20)
            edge_index = torch.Tensor(ind).long()
            edge_attr = torch.Tensor(dist)
            pos = torch.Tensor(pos)
            remission = torch.Tensor(s_data[:, 3]) / 255
            label = torch.Tensor(s_semseg).long()

            seq_num = int(self.seq_num)
            idx = torch.from_numpy(np.array([seq_num, k]))
            s_pose = torch.from_numpy(s_pose)
            data = Data(pos=pos, x=remission, y=label, edge_index=edge_index, idx=idx, car_pose=s_pose)

            data_list.append(data)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    # def del_attr(self):
    #     if 'edge_attr' in self.data.keys:
    #         del self.slices['edge_attr']
    #         self.data.edge_attr=None
    #         torch.save((self.data, self.slices), self.processed_paths[0])



if __name__ == '__main__':
    import threading
    from color import color_map
    from torch_geometric.data import DenseDataLoader
    seq_list = return_seq()
    seq_list.sort()
    for seq in seq_list[11::12]:
        print("processing ", seq)
        train_dataset = PANDASET(seq_num=seq)
        del train_dataset
        gc.collect()