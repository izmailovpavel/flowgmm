"""
Spectral Normalization from https://arxiv.org/abs/1802.05957
"""
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter
import numpy as np

def singularValues(kernel,input_shape):
    transforms = np.fft.fft2(kernel,input_shape,axes=(0,1))
    return np.linalg.svd(transforms,compute_uv=False)

#def pytorchSingularValues(kernel,input_shape):


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
        self.input_shape = x.shape[1:]
        bs = x.shape[0]
        if self._u is None:
            random_vec = torch.randn_like(x)[:1]
            self._u = random_vec/random_vec.norm()
        zeros = torch.zeros_like(self._u)
        zeros_u_and_mbx = torch.cat([zeros,self._u, x], 0)
        zeros_v_and_mby = self.module(zeros_u_and_mbx)
        bias,v_, mby = torch.split(zeros_v_and_mby,[1,1,bs])
        v = v_-bias
        self._s = s = v.norm()#torch.abs((self._u*v).sum())
        self._u = (v/v.norm()).detach()
        return mby/torch.max(s,torch.tensor(1.).to(x.device))

    def log_data(self,logger,step,name=None):
        c,h,w = self.input_shape
        weight = self.module.weight.cpu().clone().data.permute((2,3,0,1)).numpy()
        true_sigmas = singularValues(weight,(h,w)).reshape(-1)
        sigma_max = np.max(true_sigmas)
        logger.add_scalars('info',
            {f'Sigma_{name}/PowerIt':self._s.cpu().data,
             f'Sigma_{name}/True':sigma_max},step)

