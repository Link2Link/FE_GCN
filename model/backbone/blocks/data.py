# _*_ coding: utf-8 _*_
# @Time : 2020/12/28
# @Author : Chenfei Wang
# @File : data.py
# @desc :
# @note :

import torch


def pointwise_distance(x, y, square=True):
    """
    The pointwise distance from x to y
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        inner = -2 * torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.square(x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.square(y), dim=-1, keepdim=True)
        dis = x_square + inner + y_square.transpose(2, 1)
        if square:
            return dis
        else:
            return torch.sqrt(dis)

def pairwise_distance(x, y, edge_index, square=True):
    """
    The pairwise distance from x to y
    """
    with torch.no_grad():
        x_i = batched_index_select(x, edge_index[1])
        y_j = batched_index_select(y, edge_index[0])
        vec = y_j - x_i
        dis = torch.sum(torch.square(vec), dim=1)
        if square:
            return dis
        else:
            return torch.sqrt(dis)



def knn(x:torch.Tensor, y:torch.Tensor, k:int=20):
    """
    return the knn index between two data. first index is in y, second index is in x
    """
    with torch.no_grad():
        batch_size, x_dims, x_points, _ = x.shape
        batch_size, y_dims, y_points, _ = y.shape
        if x_dims != y_dims:
            raise RuntimeError('the dimension is not correct ! x_dim is {x_dim}, y_dim is {y_dim}'
                               .format(x_dim = x_dims,y_dim = y_dims))
        dis = pointwise_distance(x.detach(), y.detach(), square=True)
        _, y_idx = torch.topk(-dis, k=k)
        x_idx = torch.arange(0, x_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((y_idx, x_idx), dim=0)


def batched_index_select(x, idx):
    batch_size, num_dims, num_vertices = x.shape[:3]
    out_num_vertices = idx.shape[1]
    k = idx.shape[-1]
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices, -1)[idx, :]
    feature = feature.view(batch_size, out_num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature

class Graph:
    """
    Definition of graph data, making graph inplace and calc distance inplace
    """
    def __init__(self, pos:torch.Tensor=None, x:torch.Tensor=None, k=None, **kwargs):
        self.x = x                          # x is the feature
        self.pos = pos                      # pos is the position
        self.k = k
        for key, item in kwargs.items():
            self[key] = item
        if self.k is not None:
            self.makeGraph()

    def makeGraph(self):
        """
        using knn to build the graph
        """
        self.edge_index = knn(self.pos, self.pos, k=self.k)
        self.dis = pairwise_distance(self.pos, self.pos, self.edge_index)

    def __len__(self):
        return self.pos.shape[2]

    @property
    def feature_dim(self):
        return self.x.shape[1]

    def __str__(self):
        return "graph has %d node, %d features, %dx%d edges" % (len(self), self.feature_dim, len(self), self.k)





# if __name__ == '__main__':
#     points_1 = torch.ones(2,3,2,1)
#     points_1[0, :, 0, :] = torch.Tensor([[0.1], [0.1], [0.1]])
#     points_1[0, :, 1, :] = torch.Tensor([[0.2], [0.2], [0.2]])
#     points_2 = torch.ones(2,3, 1,1)*0.11
#     featurex = torch.Tensor(2,16,1024,1)
#     featurey = torch.Tensor(2,16,512,1)
#     g1 = Graph(x=featurex, pos=points_1, k=1)
#     g2 = Graph(x=featurey, pos=points_2, k=1)