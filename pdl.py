import numpy
import numpy as np

import torch
import torch.nn.functional as F
import torch.cuda.amp as amp

EPS=1e-8


def projectedDistributionLoss(x, y, num_projections=1000):
    '''Projected Distribution Loss (https://arxiv.org/abs/2012.09289)
    x.shape = B,M,N,...
    '''
    def rand_projections(dim, device=torch.device('cpu'), num_projections=1000):
        projections = torch.randn((dim,num_projections), device=device)
        projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=0, keepdim=True))    # columns are unit length normalized
        return projections
    def reduce(x, reduction="mean"):
        """Batch reduction of a tensor."""
        if reduction == "sum":
            x = x.sum()
        elif reduction == "mean":
            x = x.mean()
        elif reduction == "none":
            x = x
        else:
            raise ValueError("unkown reduction={}.".format(reduction))
        return x
    def huber_loss(pred, target, delta=1e-3, reduction="mean"):
        """Computes the Huber loss."""
        diff = pred - target
        abs_diff = torch.abs(diff)
        with amp.autocast(enabled=False):
            loss = torch.where(abs_diff < delta,
                               0.5 * (diff+EPS)**2,
                               delta * (abs_diff - 0.5 * delta))
        loss = reduce(loss, reduction=reduction)
        return loss
    x = x.reshape(x.shape[0], x.shape[1], -1)   # B,N,M
    y = y.reshape(y.shape[0], y.shape[1], -1)
    W = rand_projections(x.shape[-1], device=x.device, num_projections=num_projections)#x.shape[-1])
#    W = torch.repeat_interleave(W.unsqueeze(0), repeats=x.shape[0], axis=0) # B,M,M' whereM'==M
#    e_x = torch.bmm(x, W)   # B,N,M'
#    e_y = torch.bmm(y, W)
    e_x = torch.matmul(x,W) # multiplication via broad-casting
    e_y = torch.matmul(y,W)
    loss = 0
    for ii in range(e_x.shape[2]):
#        g = torch.sort(e_x[:,:,ii],dim=1)[0] - torch.sort(e_y[:,:,ii],dim=1)[0]; print(g.mean(), g.min(), g.max())
        loss = loss + F.l1_loss(torch.sort(e_x[:,:,ii],dim=1)[0] , torch.sort(e_y[:,:,ii],dim=1)[0])    # if this gives issues; try Huber loss later
    return loss
