import math
import numpy as np
import torch
from torch import nn
from torch import distributions


class RealNVP(nn.Module):
    def __init__(self, nets, nett, masks, prior, device=None):
        super().__init__()

        self.prior = prior
        self.mask = nn.Parameter(masks, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])

        self.to(device)
        self.device = device

    def g(self, z):
        x = z
        for i in reversed(range(len(self.mask))):
            mx = self.mask[i] * x
            tmx = self.t[i](mx)
            smx = self.s[i](mx)
            x = mx + (1 - self.mask[i]) * ((x - tmx) * torch.exp(-smx))
        return x

    def f(self, x):
        z = x
        log_det_J = 0
        for i in range(len(self.mask)):
            mz = self.mask[i] * z
            smz = self.s[i](mz)
            tmz = self.t[i](mz)
            z = mz + (1 - self.mask[i]) * (z * torch.exp(smz) + tmz)
            if x.dim() == 2:
                log_det_J += (smz * (1-self.mask[i])).sum(1)
            else:
                log_det_J += (smz * (1-self.mask[i])).sum(1, 2, 3)
        return z, log_det_J

    def log_prob(self, x, *args, **kwargs):
        z, log_det_J = self.f(x)
        return self.prior.log_prob(z, *args, **kwargs) + log_det_J

    def sample(self, bs=1):
        z = self.prior.sample(torch.Size([bs]))
        x = self.g(z)
        return x
