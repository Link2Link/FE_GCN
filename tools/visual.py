# _*_ coding: utf-8 _*_
# @Time : 2020/12/24
# @Author : Chenfei Wang
# @File : visual.py
# @desc :
# @note :
import numpy as np
from visualization.visualize import visualization
from matplotlib import pyplot as plt
import sys
import vispy
class visual(visualization):
    def __init__(self, data_dict):
        super(visual, self).__init__()
        self.data_dict = data_dict
        self.offset = 0
        self.draw()

    def draw(self):
        points = self.data_dict[self.offset]['points']
        if not isinstance(points, np.ndarray):
            points = points.cpu().numpy()

        self.draw_points(points, flag=0)

    def press_N(self):
        self.offset += 1 if self.offset < len(self.data_dict)-1 else 0
        self.draw()

    def press_B(self):
        self.offset -= 1 if self.offset > 0 else 0
        self.draw()


if __name__ == '__main__':
    data_dict_list = np.load('point_topo.npy', allow_pickle=True)
    # v = visual(data_dict_list)
    # if sys.flags.interactive != 1:
    #     vispy.app.run()
    data = data_dict_list[0]
    points = data['points']
    topo = data['topo']
    plt.figure()
    p1 = points[topo][:, 0, :]
    p2 = points[topo][:, 1:, :].transpose([1,0,2])
    dif = p1 - p2
    dis = np.sqrt(np.sum(np.square(dif), axis=2).transpose(1,0))
    all_dis = dis.flatten()
    # plt.hist(all_dis, 200)

    f = 2/(1+np.exp(all_dis))
    plt.hist(f, 200)


