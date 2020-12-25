# _*_ coding: utf-8 _*_
# @Time : 2020/12/25
# @Author : Chenfei Wang
# @File : graph.py
# @desc :
# @note :

import torch
from .nn import batched_index_select
import copy

class TopoGraph(object):
    """"
    x should be [bs, feature, num_points, 1]
    pos should be [bs, 3, num_points, 1]
    edge_index should be [edge/self, bs, num_points, index]
    """
    def __init__(self, x=None, edge_index=None, pos=None, edge_dis=None, y=None, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.edge_dis = edge_dis
        self.y = y
        self.pos = pos
        for key, item in kwargs.items():
            self[key] = item
        if edge_index is not None and edge_index.dtype != torch.long:
            raise ValueError(
                (f'Argument `edge_index` needs to be of type `torch.long` but '
                 f'found type `{edge_index.dtype}`.'))
        self.calc_dis()

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __len__(self):
        r"""Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    @property
    def num_nodes(self):
        bs, features, nodes, _ = self.x.shape
        return nodes

    @property
    def num_batch_size(self):
        bs, features, nodes, _ = self.x.shape
        return bs

    @property
    def num_features(self):
        bs, features, nodes, _ = self.x.shape
        return features

    def calc_dis(self):
        pos, edge_index = self.pos, self.edge_index
        pos_i = batched_index_select(pos, edge_index[1])
        pos_j = batched_index_select(pos, edge_index[0])
        vec = pos_j - pos_i
        dis = torch.sqrt(torch.sum(torch.square(vec), dim=1))
        self.edge_dis = dis
