import torch
from gcn_lib.dense import BasicConv, GraphConv2d, PlainDynBlock2d, ResDynBlock2d, ResBlock2d
from torch.nn import Sequential as Seq


class DenseDeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DenseDeepGCN, self).__init__()
        channels = opt.n_filters
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        conv = opt.conv
        c_growth = channels
        g_features = 128
        self.n_blocks = opt.n_blocks

        self.head = GraphConv2d(opt.in_channels, channels, conv, act, norm, bias)
        self.backbone = Seq(*[ResBlock2d(channels, conv, act, norm, bias)
                              for i in range(self.n_blocks - 1)])

        fusion_dims = int(channels + c_growth * (self.n_blocks - 1))

        self.fusion_block = BasicConv([fusion_dims, g_features], act, norm, bias)
        self.prediction = Seq(*[BasicConv([fusion_dims+g_features, 128], act, norm, bias),
                                BasicConv([128, 64], act, norm, bias),
                                torch.nn.Dropout(p=opt.dropout),
                                BasicConv([64, opt.n_classes], None, None, bias)])

    def forward(self, inputs, edge_index):
        num_k = edge_index.shape[2]
        self_index = torch.unsqueeze(edge_index[:, :, 0], 2).repeat(1, 1, num_k)
        gcn_index = torch.stack([edge_index, self_index])
        feats = [self.head(inputs, gcn_index)]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1], gcn_index))
        feats = torch.cat(feats, dim=1)
        fusion = torch.max_pool2d(self.fusion_block(feats), kernel_size=[feats.shape[2], feats.shape[3]])
        fusion = torch.repeat_interleave(fusion, repeats=feats.shape[2], dim=2)
        mix_feature = torch.cat((fusion, feats), dim=1)
        return self.prediction(mix_feature).squeeze(-1)


if __name__ == "__main__":
    import random, numpy as np, argparse
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = 2
    N = 1024
    device = 'cuda'

    parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN For semantic segmentation')
    parser.add_argument('--in_channels', default=9, type=int, help='input channels (default:9)')
    parser.add_argument('--n_classes', default=13, type=int, help='num of segmentation classes (default:13)')
    parser.add_argument('--k', default=20, type=int, help='neighbor num (default:16)')
    parser.add_argument('--block', default='res', type=str, help='graph backbone block type {plain, res, dense}')
    parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
    parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
    parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
    parser.add_argument('--bias', default=True, type=bool, help='bias of conv layer True or False')
    parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
    parser.add_argument('--n_blocks', default=7, type=int, help='number of basic blocks')
    parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')
    parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
    parser.add_argument('--stochastic', default=False, type=bool, help='stochastic for gcn, True or False')
    args = parser.parse_args()

    pos = torch.rand((batch_size, N, 3), dtype=torch.float).to(device)
    x = torch.rand((batch_size, N, 6), dtype=torch.float).to(device)

    inputs = torch.cat((pos, x), 2).transpose(1, 2).unsqueeze(-1)

    # net = DGCNNSegDense().to(device)
    net = DenseDeepGCN(args).to(device)
    print(net)
    out = net(inputs)
    print(out.shape)



# import random, numpy as np, argparse
# seed = 0
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
#
# batch_size = 3
# N = 1024*12
# device = 'cuda'
#
# parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN For semantic segmentation')
# parser.add_argument('--in_channels', default=9, type=int, help='input channels (default:9)')
# parser.add_argument('--n_classes', default=13, type=int, help='num of segmentation classes (default:13)')
# parser.add_argument('--k', default=20, type=int, help='neighbor num (default:16)')
# parser.add_argument('--block', default='res', type=str, help='graph backbone block type {plain, res, dense}')
# parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
# parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
# parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
# parser.add_argument('--bias', default=True, type=bool, help='bias of conv layer True or False')
# parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
# parser.add_argument('--n_blocks', default=7, type=int, help='number of basic blocks')
# parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')
# parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
# parser.add_argument('--stochastic', default=False, type=bool, help='stochastic for gcn, True or False')
# args = parser.parse_args()
#
# pos = torch.rand((batch_size, N, 3), dtype=torch.float).to(device)
# x = torch.rand((batch_size, N, 6), dtype=torch.float).to(device)
#
# inputs = torch.cat((pos, x), 2).transpose(1, 2).unsqueeze(-1)
# knn = DenseDilatedKnnGraph(k=20, dilation=1, stochastic=False, epsilon=0.2)
# net = DenseDeepGCN(args).to(device)
# # print(net)
# # out = net(inputs)
# # print(out.shape)
# with torch.autograd.profiler.profile() as prof:
#     for _ in range(10):
#         out = net(inputs)
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))
# print(prof.total_average())
