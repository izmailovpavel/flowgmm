import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
#from torch.nn.utils import spectral_norm
import scipy as sp
import scipy.sparse
from .normalizations import pad_circular_nd
from ..utils import export, conv2d

@export
class iConv2d(nn.Module):
    """ wraps conv2d in a module with an inverse function """
    def __init__(self,*args,inverse_tol=1e-7,circ=True,**kwargs):
        super().__init__()
        self.conv = conv2d(*args,**kwargs)
        self.inverse_tol = inverse_tol
        self._reverse_iters = 0
        self._inverses_evaluated = 0
        self._circ= circ
    @property
    def iters_per_reverse(self):
        return self._reverse_iters/self._inverses_evaluated
    def forward(self,x):
        self._shape = x.shape
        if self._circ:
            padded_x = pad_circular_nd(x,1,dim=[2,3])
            return F.conv2d(padded_x,self.conv.weight,self.conv.bias)
        else:
            return self.conv(x)
    # FFT inverse method
    def inverse(self,y):
        x = inverse_fft_conv3x3(y-self.conv.bias[None,:,None,None],self.conv.weight)
        # if torch.isnan(x).any():
        #     assert False, "Nans encountered in iconv2d"
        return x
@export
class iConv1x1(nn.Conv2d):
    def __init__(self, channels):
        super().__init__(channels,channels,1)

    def logdet(self):
        bs,c,h,w = self._input_shape
        return (torch.slogdet(self.weight[:,:,0,0])[1]*h*w).expand(bs)
    def inverse(self,y):
        bs,c,h,w = self._input_shape
        inv_weight = torch.inverse(self.weight[:,:,0,0].double()).float().view(c, c, 1, 1)
        debiased_y = y - self.bias[None,:,None,None]
        x = F.conv2d(debiased_y,inv_weight)
        # if torch.isnan(x).any():
        #     assert False, "Nans encountered in iconv1x1"
        return x

    def forward(self, x):
        self._input_shape = x.shape
        return F.conv2d(x,self.weight,self.bias)
