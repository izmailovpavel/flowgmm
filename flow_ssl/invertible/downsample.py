import torch
import torch.nn as nn
#from oil.utils.utils import Expression,export,Named
import torch.nn.functional as F
import numpy as np
from ..utils import export
#https://github.com/rtqichen/ffjord/blob/master/lib/layers/squeeze.py


@export
class SqueezeLayer(nn.Module):
    def __init__(self, downscale_factor=2):
        super().__init__()
        self.downscale_factor = downscale_factor
    def forward(self, x):
        return squeeze(x,self.downscale_factor)
    def inverse(self,y):
        return unsqueeze(y,self.downscale_factor)
    def logdet(self):
        return 0

def unsqueeze(input, upscale_factor=2):
    '''
    [:, C*r^2, H, W] -> [:, C, H*r, W*r]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels // (upscale_factor**2)

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(batch_size, out_channels, upscale_factor, upscale_factor, in_height, in_width)

    output = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


def squeeze(input, downscale_factor=2):
    '''
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels * (downscale_factor**2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor
    )

    output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)

def except_every4(a):
    bs,c,h,w = a.shape
    assert not c%4, "channels not divisible by 4"
    without = (a.reshape(bs,c//4,4,h,w)[:,:,1:])
    removed = without.reshape(bs,c//4 * 3,h,w)
    return removed

def add_zeros_every4(a):
    bs,c,h,w = a.shape
    assert not c%3, "Channels not divisible by 3"
    with_zeros_extended = torch.zeros(bs,c//3,4,h,w).to(a.device)
    with_zeros_extended[:,:,1:] = a.reshape(bs,c//3,3,h,w)
    return with_zeros_extended.reshape(bs,4*c//3,h,w)

def add_minus_sum_every4(a):
    bs,c,h,w = a.shape
    assert not c%3, "Channels not divisible by 3"
    with_minus_sum = torch.zeros(bs,c//3,4,h,w).to(a.device)
    a_reshaped = a.reshape(bs,c//3,3,h,w)
    with_minus_sum[:,:,1:] = a_reshaped
    with_minus_sum[:,:,0] = - a_reshaped.sum(2)
    return with_minus_sum.reshape(bs,4*c//3,h,w)

@export
class NNdownsample(nn.Module):
    def forward(self,x):
        downsampled = F.interpolate(x,scale_factor=1/2)
        resampled = F.interpolate(downsampled,scale_factor=2)
        lost_info = squeeze(x-resampled,2)
        nonzero_info = except_every4(lost_info) # channels 0,3,7,... are all zero
        return torch.cat((downsampled,nonzero_info),dim=1)
    def inverse(self,y):
        c = y.shape[1]
        downsampled,nonzero_info = torch.split(y,(c//4,3*c//4),dim=1)
        lost_info = add_zeros_every4(nonzero_info)
        nn_upsampled = F.interpolate(downsampled,scale_factor=2)
        full_upsampled = nn_upsampled + unsqueeze(lost_info)
        return full_upsampled
    def logdet(self):
        return 0

@export
class iAvgPool2d(nn.Module):
    def forward(self,x):
        self._x_shape = x.shape
        self._x_device = x.device
        downsampled = F.avg_pool2d(x,2,stride=2)
        resampled = F.interpolate(downsampled,scale_factor=2)
        lost_info = squeeze(x-resampled,2)
        nonzero_info = except_every4(lost_info) # no extra info in channels 0,4,8...
        return torch.cat((downsampled,nonzero_info),dim=1)
    def inverse(self,y):
        c = y.shape[1]
        downsampled,nonzero_info = torch.split(y,(c//4,3*c//4),dim=1)
        lost_info = add_minus_sum_every4(nonzero_info)
        avg_upsampled = F.interpolate(downsampled,scale_factor=2)# the average
        full_upsampled = avg_upsampled + unsqueeze(lost_info)
        return full_upsampled
    def logdet(self):
        bs,c,h,w = self._x_shape
        return (torch.log(torch.Tensor([1./4]))*c*h*w/4).to(self._x_device).expand(bs)

@export
class padChannels(nn.Module):
    def __init__(self, pad_size):
        super().__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]
    def logdet(self):
        return 0

@export
class RandomPadChannels(nn.Module):
    def __init__(self,pad_size):
        super().__init__()
        self.pad_size = pad_size
    def forward(self,x):
        bs = x.shape[0]
        noise = torch.randn(bs,self.pad_size,*x.shape[2:]).to(x.device)
        self._noise_shape = noise.shape
        self._device = x.device
        padded_x = torch.cat([x,noise],dim=1)
        return padded_x
    def inverse(self,x):
        return x[:,:x.size(1)-self.pad_size]
    def logdet(self):
        # entropy of the noise, coming from the variational
        # upper bound on the negative log likelihood
        d = np.prod(self._noise_shape[1:])
        bs = self._noise_shape[0]
        gaussian_entropy = d*0.5*torch.log(torch.Tensor([2*np.pi*np.e])).expand(bs).to(self._device)
        return gaussian_entropy

@export
class keepChannels(nn.Module):
    def __init__(self,k):
        """k represents the number of channels in x to keep"""
        super().__init__()
        self.k = k
    def forward(self,inp):
        x,z = inp
        x_new,z_extra = split(x,self.k)
        z.append(z_extra)
        return x_new,z
    def inverse(self,output):
        x_small,z_large = output
        #z_extra = z_large.pop(-1)
        x,z_large = merge(x_small,z_large[-1]),z_large[:-1]
        #x = merge(x_small,z_extra)
        return x, z_large
    def logdet(self):
        return 0

def split(x,k):
    x1 = x[:, :k, :, :].contiguous()
    x2 = x[:, k:, :, :].contiguous()
    return x1, x2

def merge(x1, x2):
    return torch.cat((x1, x2), 1)

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
class iLogits(nn.Module):
    def __init__(self,cnstr=0.90):# should change to .95
        super().__init__()
        self.cnstr=cnstr
    def forward(self,x):
        # assumes x values are between 0 and 1
        z = (x * 255. + torch.rand_like(x)) / 256.
        z = (2 * z - 1) * self.cnstr
        z = (z + 1) / 2
        z = z.log() - (1-z).log()
        self._z = z
        if torch.isnan(x).any():
            assert False, "Nans encountered in iLogits"
        return z
    def inverse(self,y):
        return torch.sigmoid(y)
    def logdet(self):
        y = self._z
        spl = F.softplus(torch.Tensor([1-self.cnstr]).log()-torch.Tensor([self.cnstr]).log()).to(y.device)
        logdet_output =  (F.softplus(y)+F.softplus(-y)-spl).sum(3).sum(2).sum(1)
        #print(f"ilogits_logdet_shape {logdet_output}")
        return logdet_output
