"""
Spectral Normalization from https://arxiv.org/abs/1802.05957
"""
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter

def batchwise_l2normalize(x):
    return x/x.pow(2).sum(-1).sum(-1).sum(-1).sqrt()[:,None,None,None]

class SN(nn.Module):
    def __init__(self,module,n_power_iterations=1,eps=1e-12):
        super().__init__()
        self.module = module
        assert isinstance(module,(nn.Conv1d,nn.Conv2d,nn.Conv3d)), "SN can only be applied to linear modules such as conv and linear"
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.register_buffer('_u',None)
        self.register_buffer('_s',torch.tensor(1.))

    def forward(self,x):
        if not self.training: return self.module(x)/torch.max(self._s,torch.tensor(1.).to(x.device))
        bs = x.shape[0]
        if self._u is None:
            random_vec = torch.randn_like(x)[:1]
            self._u = random_vec/random_vec.norm()
        zeros = torch.zeros_like(self._u)
        zeros_u_and_mbx = torch.cat([zeros,self._u, x], 0)
        zeros_v_and_mby = self.module(zeros_u_and_mbx)
        bias,v_, mby = torch.split(zeros_v_and_mby,[1,1,bs])
        v = v_.detach()-bias
        self._s = torch.abs((self._u*v).sum())
        self._u = (v/v.norm()).detach()
        return mby/torch.max(self._s,torch.tensor(1.).to(x.device))