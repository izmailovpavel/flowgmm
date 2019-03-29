import torch
import torch.nn as nn
from oil.utils.utils import Expression,export,Named
#https://github.com/rtqichen/ffjord/blob/master/lib/layers/squeeze.py


@export
class SqueezeLayer(nn.Module):
    def __init__(self, downscale_factor):
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
        z_extra = z_large.pop(-1)
        x = merge(x_small,z_extra)
        return x, z_large
    def logdet(self):
        return 0

def split(x,k):
    x1 = x[:, :k, :, :].contiguous()
    x2 = x[:, k:, :, :].contiguous()
    return x1, x2

def merge(x1, x2):
    return torch.cat((x1, x2), 1)