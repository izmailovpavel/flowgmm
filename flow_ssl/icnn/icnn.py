import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
import numpy as np
from ..utils import export, Named, Expression
from ..conv_parts import ResBlock,conv2d
from ..invertible import SqueezeLayer,split,merge,padChannels,keepChannels,NNdownsample,iAvgPool2d
from ..invertible import iLogits, iBN, MeanOnlyBN, iSequential, passThrough, addZslot, Join, pad_circular_nd
from ..invertible import  iConv2d, iSLReLU,iConv1x1,Flatten,RandomPadChannels
import scipy as sp
import scipy.sparse


def iConvSelu(channels):
    return iSequential(iConv2d(channels,channels),iSLReLU())

def iConvBNselu(channels):
    return iSequential(iConv2d(channels,channels),iBN(channels),iSLReLU())

def StandardNormal(d):
    return Independent(Normal(0,1),d)

class FlowNetwork(nn.Module,metaclass=Named):
    def forward(self,x):
        return self.classifier_head(self.body(x))
    def sample(self,bs=1):
        return self.flow.inverse(self.prior.sample([bs]).to(self.device))
    @property
    def device(self):
        try: return self._device
        except AttributeError:
            self._device = next(self.parameters()).device
            return self._device
    def nll(self,x):
        z = self.flow(x)
        logdet = self.flow.logdet()
        return  -1*self.prior.log_prob(z) - logdet

class iCNN(FlowNetwork):
    """
    Very small CNN
    """
    def __init__(self, num_classes=10,k=16):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.body = iSequential(
            iLogits(),
            RandomPadChannels(k-3),
            *iConvBNselu(k),
            *iConvBNselu(k),
            *iConvBNselu(k),
            NNdownsample(),
            *iConvBNselu(4*k),
            *iConvBNselu(4*k),
            *iConvBNselu(4*k),
            NNdownsample(),
            *iConvBNselu(16*k),
            *iConvBNselu(16*k),
            *iConvBNselu(16*k),
            iConv2d(16*k,16*k),
        )
        self.classifier_head = nn.Sequential(
            nn.BatchNorm2d(16*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(16*k,num_classes)
        )
        self.flow = iSequential(self.body,Flatten())
        self.prior = StandardNormal(k*32*32)


class MultiScaleiCNN(iCNN):
    def __init__(self, num_classes=10,k=32):
        super().__init__(num_classes,k)
        self.num_classes = num_classes
        self.k = k
        self.body = iSequential(
            iLogits(),
            RandomPadChannels(k-3),
            addZslot(),

            passThrough(*iConvBNselu(k)),
            passThrough(*iConvBNselu(k)),
            passThrough(*iConvBNselu(k)),
            passThrough(NNdownsample()),
            passThrough(iConv1x1(4*k)),
            keepChannels(2*k),
            
            passThrough(*iConvBNselu(2*k)),
            passThrough(*iConvBNselu(2*k)),
            passThrough(*iConvBNselu(2*k)),
            passThrough(NNdownsample()),
            passThrough(iConv1x1(8*k)),
            keepChannels(4*k),
            
            passThrough(*iConvBNselu(4*k)),
            passThrough(*iConvBNselu(4*k)),
            passThrough(*iConvBNselu(4*k)),
            passThrough(iConv2d(4*k,4*k)),
            Join(),
        )
        self.classifier_head = nn.Sequential(
            Expression(lambda z:z[-1]),
            nn.BatchNorm2d(4*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(4*k,num_classes)
        )
        self.flow = iSequential(self.body,Flatten())
        self.prior = StandardNormal(k*32*32)

class MultiScaleiCNNv2(MultiScaleiCNN):
    def __init__(self, num_classes=10,k=128):
        super().__init__(num_classes,k)
        self.num_classes = num_classes
        self.k = k
        self.body = iSequential(
            iLogits(),
            RandomPadChannels(k-3),
            addZslot(),

            passThrough(*iConvBNselu(k)),
            passThrough(*iConvBNselu(k)),
            passThrough(*iConvBNselu(k)),
            passThrough(NNdownsample()),
            passThrough(iConv1x1(4*k)),
            keepChannels(2*k),
            
            passThrough(*iConvBNselu(2*k)),
            passThrough(*iConvBNselu(2*k)),
            passThrough(*iConvBNselu(2*k)),
            passThrough(NNdownsample()),
            passThrough(iConv1x1(8*k)),
            keepChannels(4*k),
            
            passThrough(*iConvBNselu(2*k)),
            passThrough(*iConvBNselu(2*k)),
            passThrough(*iConvBNselu(2*k)),
            passThrough(iConv2d(2*k,2*k)),
            Join(),
        )
        self.classifier_head = nn.Sequential(
            Expression(lambda z:z[-1]),
            nn.BatchNorm2d(2*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(2*k,num_classes)
        )
        self.flow = iSequential(self.body,Flatten())
        self.prior = StandardNormal(k*32*32)

class iCNN3d(FlowNetwork):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.body = iSequential(
            iLogits(),
            *iConvBNselu(3),
            *iConvBNselu(3),
            *iConvBNselu(3),
            NNdownsample(),
            *iConvBNselu(12),
            *iConvBNselu(12),
            *iConvBNselu(12),
            NNdownsample(),
            *iConvBNselu(48),
            *iConvBNselu(48),
            *iConvBNselu(48),
            NNdownsample(),
            *iConvBNselu(192),
            *iConvBNselu(192),
            *iConvBNselu(192),
            iConv2d(192,192),
        )
        self.classifier_head = nn.Sequential(
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(192,num_classes)
        )
        self.flow = iSequential(self.body,Flatten())
        self.prior = StandardNormal(3*32*32)


class iLinear3d(iCNN3d):

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.body = iSequential(
            iLogits(),
            iConv2d(3,3),
            iConv2d(3,3),
            iConv2d(3,3),
            NNdownsample(),
            iConv2d(12,12),
            iConv2d(12,12),
            iConv2d(12,12),
            NNdownsample(),
            iConv2d(48,48),
            iConv2d(48,48),
            iConv2d(48,48),
            NNdownsample(),
            iConv2d(192,192),
            iConv2d(192,192),
            iConv2d(192,192),
        )
        self.classifier_head = nn.Sequential(
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(192,num_classes)
        )
        self.flow = iSequential(self.body,Flatten())
        self.prior = StandardNormal(3*32*32)