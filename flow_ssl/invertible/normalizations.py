
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from ..utils import export
@export
class MeanOnlyBN(nn.BatchNorm2d):
    # See https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        if 'affine' in kwargs:
            assert kwargs['affine'], "only affine supported"

    def forward(self, input_):
        self._check_input_dim(input_)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
        sum_ = input_.sum(1)
        sum_of_square = input_.pow(2).sum(1)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean

        self.running_mean = (
                (1 - exponential_average_factor) * self.running_mean
                + exponential_average_factor * mean.detach()
        )
        unbias_var = sumvar / (numel - 1)
        self.running_var = (
                (1 - exponential_average_factor) * self.running_var
                + exponential_average_factor * unbias_var.detach()
        )
        bias_var = sumvar / numel
        if self.track_running_stats and not self.training:
            mean, bias_var = self.running_mean,self.running_var
        inv_std = 1 / (bias_var + self.eps).pow(0.5).unsqueeze(1)
        mul = torch.min(inv_std * self.weight.unsqueeze(1),torch.ones_like(inv_std))
        output = ((input_ - mean.unsqueeze(1))  + self.bias.unsqueeze(1))

        return output.view(channels, batchsize, height, width).permute(1, 0, 2, 3).contiguous()

@export
class iBN(nn.BatchNorm2d):
    # See https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.weight.data/= self.weight.data
        self.bias.data-=self.bias.data
        if 'affine' in kwargs:
            assert kwargs['affine'], "only affine supported"

    def forward(self, input_):
        self._check_input_dim(input_)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
        sum_ = input_.sum(1)
        sum_of_square = input_.pow(2).sum(1)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean

        self.running_mean = (
                (1 - exponential_average_factor) * self.running_mean
                + exponential_average_factor * mean.detach()
        )
        unbias_var = sumvar / (numel - 1)
        self.running_var = (
                (1 - exponential_average_factor) * self.running_var
                + exponential_average_factor * unbias_var.detach()
        )
        bias_var = sumvar / numel
        if self.track_running_stats and not self.training:
            mean, bias_var = self.running_mean,self.running_var
        inv_std = 1 / (bias_var + self.eps).pow(0.5).unsqueeze(1)
        self.numel,self.height,self.width,self.batchsize = numel,height,width,batchsize
        mul = inv_std * self.weight.unsqueeze(1)
        self.inv_std = inv_std.detach()
        output = ((input_ - mean.unsqueeze(1))*mul  + self.bias.unsqueeze(1))

        return output.view(channels, batchsize, height, width).permute(1, 0, 2, 3).contiguous()

    def logdet(self):
        mul = self.inv_std*self.weight.unsqueeze(1)
        bn_logdet = (torch.log(mul).sum()*self.height*self.width).expand(self.batchsize)
        #print(f"BN logdet: {bn_logdet}")
        return bn_logdet

    def inverse(self,y):
        assert not self.training, "inverse must be computed in eval mode"
        batchsize, channels, height, width = y.size()
        numel = batchsize * height * width
        y_reshaped = y.permute(1,0,2,3).contiguous().view(channels,numel)
        mul = self.weight.unsqueeze(1)/(self.running_var + self.eps).pow(0.5).unsqueeze(1)
        unsquashed_y = (y_reshaped - self.bias.unsqueeze(1))/mul + self.running_mean.unsqueeze(1)
        x = unsquashed_y.view(channels,batchsize,height,width).permute(1,0,2,3).contiguous()
        #assert not torch.isnan(x).any(), "Nans in iBN"
        return x


def singularValues(kernel,input_shape):
    transforms = np.fft.fft2(kernel,input_shape,axes=(0,1))
    return np.linalg.svd(transforms,compute_uv=False)

#def pytorchSingularValues(kernel,input_shape):
@export
def pad_circular_nd(x: torch.Tensor, pad: int, dim) -> torch.Tensor:
    """
    :param x: shape [H, W]
    :param pad: int >= 0
    :param dim: the dimension over which the tensors are padded
    :return:
    """
    if isinstance(dim, int):
        dim = [dim]

    for d in dim:
        if d >= len(x.shape):
            raise IndexError("dim {} out of range".format(d))

        idx = tuple(slice(0, None if s != d else pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x, x[idx]], dim=d)

        idx = tuple(slice(None if s != d else -2 * pad, None if s != d else -pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x[idx], x], dim=d)
        pass
    #print(x.shape)
    return x

@export
def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def batchwise_l2normalize(x):
    return x/x.pow(2).sum(-1).sum(-1).sum(-1).sqrt()[:,None,None,None]
@export
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
        self.input_shape = x.shape[1:]
        if not self.training: return self.module(x)/torch.max(self._s,torch.tensor(1.).to(x.device))
        
        bs = x.shape[0]
        if self._u is None:
            random_vec = torch.randn_like(x)[:1]
            self._u = random_vec/random_vec.norm()
        zeros = torch.zeros_like(self._u)
        zeros_u_and_mbx = torch.cat([zeros,self._u, x], 0)
        zeros_v_and_mby = self.module(zeros_u_and_mbx)
        bias,v_, mby = torch.split(zeros_v_and_mby,[1,1,bs])
        v = v_-bias
        #print(v.shape)
        W_T = flip(flip(self.module.weight.permute(1,0,2,3),2),3) # Transpose channels, flip filter
        u = F.conv2d(v,W_T,padding=self.module.padding)
        #print(u.shape)
        #print(self._u.shape)
        self._s = s = torch.sqrt((self._u*u).sum())
        self._u = (u/u.norm()).detach()
        return mby/torch.max(s,torch.tensor(1.).to(x.device))

    def log_data(self,logger,step,name=None):
        c,h,w = self.input_shape
        weight = self.module.weight.cpu().clone().data.permute((2,3,0,1)).numpy()
        true_sigmas = singularValues(weight,(h,w)).reshape(-1)
        sigma_max = np.max(true_sigmas)
        logger.add_scalars('info',
            {'Sigma_{}/PowerIt'.format(name):self._s.cpu().data,
             'Sigma_{}/True'.format(name):sigma_max},step)

# Ported from Residual Flows repo

class ActNormNd(nn.Module):

    def __init__(self, num_features, eps=1e-12):
        super(ActNormNd, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer('initialized', torch.tensor(0))

    @property
    def shape(self):
        raise NotImplementedError
    
    def forward(self,x):
        self._last_x_size = x.size()
        c = x.size(1)
        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_t = x.transpose(0, 1).contiguous().view(c, -1)
                batch_mean = torch.mean(x_t, dim=1)
                batch_var = torch.var(x_t, dim=1)

                # for numerical issues
                batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))

                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var))
                self.initialized.fill_(1)
        bias = self.bias.view(*self.shape).expand_as(x)
        weight = self.weight.view(*self.shape).expand_as(x)
        y = (x + bias) * torch.exp(weight)
        return y
    

    def inverse(self, y, logpy=None):
        assert self.initialized
        bias = self.bias.view(*self.shape).expand_as(y)
        weight = self.weight.view(*self.shape).expand_as(y)
        x = y * torch.exp(-weight) - bias
        return x

    def logdet(self):
        ld = self._logdetgrad(self._last_x_size)
        #print(ld.shape)
        return ld

    def _logdetgrad(self, size):
        return self.weight.view(*self.shape).expand(*size).contiguous().view(size[0], -1).sum(1, keepdim=False)

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))

@export
class ActNorm1d(ActNormNd):

    @property
    def shape(self):
        return [1, -1]

@export
class ActNorm2d(ActNormNd):

    @property
    def shape(self):
        return [1, -1, 1, 1]
