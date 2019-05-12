import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from ..utils import conv2d,export
import torchcontrib.nn.functional as contrib
from torchdiffeq import odeint_adjoint as odeint



@export
class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)
@export
class ConcatResBlock(nn.Module):

    def __init__(self, dim,gn=True,add=False):
        super().__init__()
        norm_layer = (lambda c: nn.GroupNorm(c//16,c)) if gn else nn.BatchNorm2d
        self.norm1 = norm_layer(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm_layer(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm_layer(dim)
        self.nfe = 0
        self.add = add

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        #out = self.norm3(out)# why is this here??
        if self.add: out = out+x
        return out

@export
class ConcatBottleBlock(nn.Module):
    def __init__(self,channels,gn=True,add=False):
        super().__init__()
        norm_layer = (lambda c: nn.GroupNorm(c//16,c)) if gn else nn.BatchNorm2d
        self.norm1 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(channels, channels//4, 1, padding=0)
        self.norm2 = norm_layer(channels//4)
        self.conv2 = ConcatConv2d(channels//4, channels//4, 3, 1, 1)
        self.norm3 = norm_layer(channels//4)
        self.conv3 = ConcatConv2d(channels//4, channels, 1, padding=0)
        self.nfe = 0
        self.add = add

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(t,out)
        #out = self.norm3(out)# why is this here??
        if self.add: out = out+x
        return out

@export
class ODEBlock(nn.Module):

    def __init__(self, odefunc,tol=3e-3):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.tol=tol

    def forward(self, x):
        self.nfe=0
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
@export
class RNNBlock(nn.Module):
    def __init__(self,module,L=10):
        super().__init__()
        self.module = module
        self.L=L
        self.times = torch.linspace(0,1,self.L)
    def forward(self,x):
        for t in self.times.type_as(x):
            x = self.module(t,x)
        return x

# Curves. see https://github.com/timgaripov/dnn-mode-connectivity/blob/master/curves.py
from scipy.special import binom
from torch.nn.modules.utils import _pair
import numpy as np
import torch.nn.init as init
import math

class Bezier(nn.Module):
    def __init__(self, num_bends):
        super().__init__()
        self.register_buffer(
            'binom',
            torch.Tensor(binom(num_bends - 1, np.arange(num_bends), dtype=np.float32))
        )
        self.register_buffer('range', torch.arange(0, float(num_bends)))
        self.register_buffer('rev_range', torch.arange(float(num_bends - 1), -1, -1))

    def forward(self, t):
        return self.binom * \
               torch.pow(t, self.range) * \
               torch.pow((1.0 - t), self.rev_range)


class BezierConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,bends, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, bias)
        kernel_size = _pair(kernel_size)      
        self.weight = Parameter(torch.Tensor(bends,
                out_channels, in_channels // groups, *kernel_size))
        if bias: self.bias = Parameter(torch.Tensor(bends,out_channels))
        self.reset_parameters()
        self.bezier = Bezier(bends)

    def reset_parameters(self):
        n = self.in_channels
        for weight in self.weight:
            init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.bias is not None:
            for bias in self.bias:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[0])
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(bias, -bound, bound)

    def forward(self,t,x):
        coeffs = self.bezier(t)
        weights = (coeffs[:,None,None,None,None]*self.weight).sum(0)
        bias = (coeffs[:,None]*self.bias).sum(0)
        return F.conv2d(x, weights, bias, self.stride,
                        self.padding, self.dilation, self.groups)

# Will we need regularization to keep the weights similar?

@export
class BezierResBlock(nn.Module):

    def __init__(self, dim,gn=True,add=False,bends=10):
        super().__init__()
        norm_layer = (lambda c: nn.GroupNorm(c//16,c)) if gn else nn.BatchNorm2d
        self.norm1 = norm_layer(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = BezierConv2d(dim, dim, 3,bends, 1, 1)
        self.norm2 = norm_layer(dim)
        self.conv2 = BezierConv2d(dim, dim, 3,bends, 1, 1)
        self.norm3 = norm_layer(dim)
        self.nfe = 0
        self.add = add

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        #out = self.norm3(out)# why is this here??
        if self.add: out = out+x
        return out