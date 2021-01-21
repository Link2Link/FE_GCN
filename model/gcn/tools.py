import torch

def index_select(x, idx):
    """
    input :     x -> [N, F, 1]     F is feature dimension
                batch_idx -> [N]
                idx -> [M, K]
    output :    selected -> [M, F, K]
    """
    N, F = x.shape[:2]
    M, K = idx.shape
    idx = idx.contiguous().view(-1)
    selected = x.transpose(2, 1)[idx]
    selected = selected.view(M, K, F).permute(0, 2, 1).contiguous()
    return selected

def pointwise_distance(x, y, square=True):
    """
    The pointwise distance from x to y
    """
    with torch.no_grad():
        x = x.squeeze(-1)
        y = y.squeeze(-1)

        x = x.unsqueeze(-1)
        y = y.transpose(0,1).unsqueeze(0)
        diff = x - y
        dis = torch.sum(torch.square(diff), dim=1)
        if torch.min(dis) < 0:
            raise RuntimeError('dis small than 0')
        if square:
            return dis
        else:
            return torch.sqrt(dis)

def knn(x, batch_idx, k:int=16):
    """
    input :     x -> [N, F, 1]
                batch_idx -> [N]
                k -> int
    output :    index -> [N, K]
    """
    with torch.no_grad():
        batch_size = torch.max(batch_idx) + 1
        index_base = torch.zeros([x.shape[0], 1], dtype=torch.long, device=x.device)
        index_list = []
        base = 0
        for bs in range(batch_size):
            x_bs = x[batch_idx==bs]
            dis = pointwise_distance(x_bs.detach(), x_bs.detach())
            _, idx = torch.topk(-dis, k=k)
            index_list.append(idx)
            index_base[batch_idx==bs] = base
            base += len(x_bs)
        index = torch.cat(index_list, dim=0) + index_base
    return index