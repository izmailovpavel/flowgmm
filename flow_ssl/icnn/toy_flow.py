import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from oil.utils.utils import Expression,export,Named
from oil.architectures.parts import ResBlock,conv2d
from .downsample import SqueezeLayer,split,merge,padChannels,keepChannels,NNdownsample,iAvgPool2d
from .clipped_BN import MeanOnlyBN, iBN
#from torch.nn.utils import spectral_norm
from .auto_inverse import iSequential
import scipy as sp
import scipy.sparse
from .iresnet import both, I, addZslot, Join,flatten
from .spectral_norm import pad_circular_nd
from .iEluNetwork import iSLReLU



class iLinear(nn.Linear):
    def __init__(self, channels):
        super().__init__(channels,channels,1)

    def logdet(self):
        bs,c = self._input_shape
        return torch.slogdet(self.weight)[1].expand(bs)
    def inverse(self,y):
        bs,c = self._input_shape
        inv_weight = torch.inverse(self.weight.double()).float()
        debiased_y = y - self.bias
        x = F.linear(debiased_y,inv_weight)
        # if torch.isnan(x).any():
        #     assert False, "Nans encountered in iconv1x1"
        return x

    def forward(self, x):
        self._input_shape = x.shape
        return super().forward(x)

class padChannels1d(nn.Module):
    def __init__(self,pad_size):
        self.pad_size = pad_size
        super().__init__()
    def forward(self,x):
        return F.pad(x,(0,self.pad_size))
    def inverse(self,x):
        return x[:,:x.size(1)-self.pad_size]
    def logdet(self):
        return 0

class iToy(nn.Module,metaclass=Named):
    """
    Toy invertible network for 2d data
    """
    def __init__(self, dim=2,k=64):
        super().__init__()
        self.dim=dim
        self.k = k
        self.body = iSequential(
            #padChannels1d(k-dim),
            iLinear(k),iSLReLU(.01),
            iLinear(k),iSLReLU(.01),
            iLinear(k),iSLReLU(.01),
            iLinear(k),iSLReLU(.01),
            iLinear(k),
        )
    def forward(self,x):
        bs,c = x.shape
        #extra_noise = 1*torch.randn(bs,self.k-self.dim)
        #z = self.body(torch.cat((x,extra_noise),dim=1))
        return self.body(x)#z
    def logdet(self):
        return self.body.logdet()
    def inverse(self,z):
        return self.body.inverse(z)[:,:self.dim]
    def prior_nll(self,z):
        d = z.shape[1]
        weighting = torch.ones(d)
        #weighting[self.dim:]*=5
        return (.5*(z*z)*weighting).sum(1) + .5*np.log(2*np.pi)*d

    def nll(self,x):
        z = self(x)
        logdet = self.logdet()
        return  self.prior_nll(z) - logdet
    def sample(self,bs=1):
        z_all = torch.randn(bs,self.k).to(self.device)
        #z_all = torch.cat((torch.randn(bs,self.dim),torch.zeros(bs,self.k-self.dim)),dim=1)
        return self.inverse(z_all)

    @property
    def device(self):
        try: return self._device
        except AttributeError:
            self._device = next(self.parameters()).device
            return self._device
