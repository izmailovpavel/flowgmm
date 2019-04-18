import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from oil.utils.utils import Expression,export,Named
from oil.architectures.parts import ResBlock,conv2d
from blocks import ConcatResBlock,ODEBlock,RNNBlock
from blocks import ConcatBottleBlock,BezierResBlock
from downsample import SqueezeLayer,split,merge,padChannels,keepChannels
from clipped_BN import MeanOnlyBN, iBN
#from torch.nn.utils import spectral_norm
from auto_inverse import iSequential


class iConv2d(nn.Module):
    """ wraps conv2d in a module with an inverse function """
    def __init__(self,*args,inverse_tol=1e-7,**kwargs):
        super().__init__()
        self.conv = conv2d(*args,**kwargs)
        self.inverse_tol = inverse_tol
        self._reverse_iters = 0
        self._inverses_evaluated = 0
    @property
    def iters_per_reverse(self):
        return self._reverse_iters/self._inverses_evaluated
    def forward(self,inp):
        return self.conv(inp)
    def inverse(self,output):
        self._inverses_evaluated +=1
        raise NotImplementedError
    def logdet(self):
        raise NotImplementedError

class iElu(nn.ELU):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        self._last_x = x
        return super().forward(x)
    def inverse(self,y):
        # y if y>0 else log(1+y)
        return F.relu(y) - F.relu(-y)*torch.log(1+y)/y
    def logdet(self):
        #logdetJ = \sum_i log J_ii # sum over c,h,w not batch
        return (-F.relu(-self._last_x)).sum(3).sum(2).sum(1) 

def iConvBNelu(ch):
    return [iConv2d(ch,ch),iBN(ch),iElu()]



class iEluNet(nn.Module,metaclass=Named):
    """
    Very small CNN
    """
    def __init__(self, num_classes=10,k=32):
        super().__init__()
        self.num_classes = num_classes
        self.body = iSequential(
            padChannels(k-3),
            *iConvBNelu(k),
            *iConvBNelu(k),
            *iConvBNelu(k),
            SqueezeLayer(2),
            *iConvBNelu(4*k),
            *iConvBNelu(4*k),
            *iConvBNelu(4*k),
            SqueezeLayer(2),
            *iConvBNelu(16*k),
            *iConvBNelu(16*k),
            *iConvBNelu(16*k),
        )
        self.head = nn.Sequential(
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(16*k,num_classes)
        )
    def forward(self,x):
        return self.head(self.body(x))
    def logdet(self):
        return self.body.logdet()
    def inverse(self,z):
        return self.body.inverse(z)