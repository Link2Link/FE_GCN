import numpy as np
import random
import copy


class G:
    def __init__(self, vertex_num, topo, label):
        self.index = np.arange(vertex_num)
        self.topo = topo
        self.label = label

    def rand_vertex(self, label):
        random.seed(0)
        if any(self.label == label):
            index = self.index[self.label==label]
            return index[random.randint(0, len(index)-1)]
        else:
            return None

    def neighbour(self, index):
        if index is None:
            return None
        index = self.topo[index].reshape(-1)
        flag = np.zeros_like(self.index)
        flag[index] = 1
        return self.index[flag==1]

    def neighbour_with_label(self, index, label):
        index = self.neighbour(index)
        if index is None:
            return None
        index = index[self.label[index]==label]
        return index

    def single_cut(self, label):
        v = self.rand_vertex(label)
        if v is None:
            return None
        num_L = 1
        v_n = self.neighbour_with_label(v, label)
        num_N = len(v_n)
        while num_L != num_N:
            num_L = num_N
            v_n = self.neighbour_with_label(v_n, label)
            num_N = len(v_n)
        return v_n

    def graph_cut(self, label):
        random.seed(0)
        v_list = []
        while np.sum(self.label==label):
            v = self.single_cut(label)
            if v is None:
                break
            self.label[v] = -1
            v_list.append(v)
        self.label[self.label<0] = label
        return v_list



