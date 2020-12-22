import torch
import torch.nn.functional as F


def slicedWassersteinLoss(x, y, num_projections=1000):
    '''random projections of 1D features to compute sliced wasserstein distance
        sliced wasserstein is more memory consuming than PDL loss
    '''
    x = x.reshape(x.shape[0], -1)   # B,L
    y = y.reshape(y.shape[0], -1)
    W = torch.randn((num_projections,x.shape[1]), device=x.device)  # this may be improved by uniformly sampling from hypersphere
    W = W / torch.sqrt(torch.sum(W ** 2, dim=1, keepdim=True))  # each row is norm=1
    e_x = torch.matmul(x, W.t())    # B,N
    e_y = torch.matmul(y, W.t())
    e_x_s = torch.sort(e_x, dim=1)[0]
    e_y_s = torch.sort(e_y, dim=1)[0]
    loss = F.l1_loss(e_x_s, e_y_s)
    return loss
