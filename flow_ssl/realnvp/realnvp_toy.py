import math
import numpy as np
import torch
from torch import nn
from torch import distributions

from flow_ssl.invertible.auto_inverse import iSequential

class ToyCouplingLayer(nn.Module):

    def __init__(self, in_dim, mid_dim, num_layers, mask):
        
        super(ToyCouplingLayer, self).__init__()
        self.mask = mask
        self.nets = nn.Sequential(nn.Linear(in_dim, mid_dim),
                                 nn.ReLU(),
                                 *self._inner_seq(num_layers, mid_dim),
                                 nn.Linear(mid_dim, in_dim),
                                 nn.Tanh(),
                                 ToyRescale(in_dim))
        self.nett = nn.Sequential(nn.Linear(in_dim, mid_dim),
                                 nn.ReLU(),
                                 *self._inner_seq(num_layers, mid_dim),
                                 nn.Linear(mid_dim, in_dim))
                                 
    @staticmethod
    def _inner_seq(num_layers, mid_dim):
        res = []
        for _ in range(num_layers):
            res.append(nn.Linear(mid_dim, mid_dim))
            res.append(nn.ReLU())
        return res

    def forward(self, x):
        z = x
        mz = self.mask * z
        smz = self.nets(mz)
        tmz = self.nett(mz)
        z = mz + (1 - self.mask) * (z * torch.exp(smz) + tmz)
        self._logdet = (smz * (1-self.mask)).sum(1)
        return z

    def inverse(self, y):
        x = y
        mx = self.mask * x
        tmx = self.nett(mx)
        smx = self.nets(mx)
        x = mx + (1 - self.mask) * ((x - tmx) * torch.exp(-smx))
        return x

    def logdet(self):
        return self._logdet


class ToyRescale(nn.Module):
    def __init__(self, D):
        super(ToyRescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(D))

    def forward(self, x):
        x = self.weight * x
        return x


class ToyRealNVPBase(nn.Module):
    def __init__(self, masks, in_dim, mid_dim, num_layers):
        super().__init__()

        self.masks = nn.Parameter(masks, requires_grad=False)
        self.body = iSequential(*[
                        ToyCouplingLayer(in_dim, mid_dim, num_layers, mask)
                        for mask in self.masks
                    ])
        
    
    def forward(self, x):
        return self.body(x)

    def logdet(self):
        return self.body.logdet()

    def inverse(self, z):
        return self.body.inverse(z)


class ToyRealNVP(ToyRealNVPBase):

    def __init__(self, in_dim=2, num_coupling_layers=6, mid_dim=256, 
                 num_layers=2):
        d = in_dim // 2
        masks = torch.zeros(num_coupling_layers, in_dim)
        for i in range(masks.size(0)):
            if i % 2:
                masks[i, :d] = 1.
            else:
                masks[i, d:] = 1. 
        super(ToyRealNVP, self).__init__(masks, in_dim, mid_dim, num_layers)
