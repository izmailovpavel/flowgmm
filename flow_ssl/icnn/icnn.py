import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
import numpy as np
from ..utils import export, Named, Expression
from ..conv_parts import ResBlock,conv2d
from ..invertible import SqueezeLayer,padChannels,keepChannels,NNdownsample,iAvgPool2d#iSequential2
from ..invertible import iLogits, iBN, MeanOnlyBN, iSequential, passThrough, addZslot, Join, pad_circular_nd
from ..invertible import  iConv2d, iSLReLU,iConv1x1,Flatten,RandomPadChannels,iLeakyReLU,iCoordInjection,iSimpleCoords
import scipy as sp
import scipy.sparse


def iConvSelu(channels):
    return iSequential(iConv2d(channels,channels),iSLReLU())

def iCoordSelu(channels):
    return iSequential(iConv2d(channels,channels),iSLReLU(),iCoordInjection(channels))

def iConvBNselu(channels):
    return iSequential(iConv2d(channels,channels),iBN(channels),iSLReLU())#iSLReLU())

def StandardNormal(d,device=torch.device('cuda:0')):
    return Independent(Normal(torch.zeros(d).to(device),torch.ones(d).to(device)),1)

class FlowNetwork(nn.Module,metaclass=Named):
    def forward(self,x):
        return self.classifier_head(self.body(x))
    def sample(self,bs=1):
        return self.flow.inverse(self.prior(self.device).sample([bs]))
    @property
    def device(self):
        try: return self._device
        except AttributeError:
            self._device = next(self.parameters()).device
            return self._device
    def nll(self,x):
        z = self.flow(x)
        logdet = self.flow.logdet()
        return  -1*(self.prior(x.device).log_prob(z) + logdet)
@export
class iCNN(FlowNetwork):
    """
    Very small CNN
    """
    def __init__(self, num_classes=10,k=16):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.body = iSequential(
            #iLogits(),
            RandomPadChannels(k-3),
            *iCoordSelu(k),
            *iCoordSelu(k),
            *iCoordSelu(k),
            NNdownsample(),
            *iCoordSelu(4*k),
            *iCoordSelu(4*k),
            *iCoordSelu(4*k),
            NNdownsample(),
            *iCoordSelu(16*k),
            *iCoordSelu(16*k),
            iConv2d(16*k,16*k),
        )
        self.classifier_head = nn.Sequential(
            nn.BatchNorm2d(16*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(16*k,num_classes)
        )
        self.flow = iSequential(self.body,Flatten())
        self.prior = StandardNormal(k*32*32)

@export
class MultiScaleiCNN(iCNN):
    def __init__(self, num_classes=10,k=64):
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

@export
class MultiScaleiCNNv2(MultiScaleiCNN):
    def __init__(self, num_classes=10,k=96):
        super().__init__(num_classes,k)
        self.num_classes = num_classes
        self.k = k
        self.body = iSequential(
            #iLogits(),
            RandomPadChannels(k-3),
            addZslot(),

            passThrough(*iConvSelu(k)),
            passThrough(*iConvSelu(k)),
            passThrough(*iConvSelu(k)),
            passThrough(NNdownsample()),
            passThrough(iConv1x1(4*k)),
            keepChannels(2*k),
            
            passThrough(*iConvSelu(2*k)),
            passThrough(*iConvSelu(2*k)),
            #passThrough(*iConvSelu(2*k)),
            passThrough(NNdownsample()),
            passThrough(iConv1x1(8*k)),
            keepChannels(2*k),
            
            passThrough(*iConvSelu(2*k)),
            passThrough(*iConvSelu(2*k)),
            #passThrough(*iConvSelu(2*k)),
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

class iCNNsup(MultiScaleiCNN):
    def __init__(self, num_classes=10,k=96):
        super().__init__(num_classes,k)
        self.num_classes = num_classes
        self.k = k
        self.body = iSequential(
            #iLogits(),
            RandomPadChannels(k-3),
            addZslot(),

            passThrough(*iConvSelu(k)),
            passThrough(*iConvSelu(k)),
            passThrough(iAvgPool2d()),
            passThrough(iConv1x1(4*k)),
            keepChannels(2*k),
            
            passThrough(*iConvSelu(2*k)),
            passThrough(*iConvSelu(2*k)),
            #passThrough(*iConvSelu(2*k)),
            passThrough(iAvgPool2d()),
            passThrough(iConv1x1(8*k)),
            keepChannels(2*k),
            
            passThrough(*iConvSelu(2*k)),
            passThrough(*iConvSelu(2*k)),
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

class iSimpleSup(MultiScaleiCNN):
    def __init__(self, num_classes=10,k=96):
        super().__init__(num_classes,k)
        self.num_classes = num_classes
        self.k = k
        self.body = iSequential(
            #iLogits(),
            RandomPadChannels(k-3),
            addZslot(),

            passThrough(*iConvSelu(k)),
            passThrough(*iConvSelu(k)),
            passThrough(iAvgPool2d()),
            keepChannels(2*k),
            
            passThrough(*iConvSelu(2*k)),
            passThrough(*iConvSelu(2*k)),
            #passThrough(*iConvSelu(2*k)),
            passThrough(iAvgPool2d()),
            keepChannels(2*k),
            
            passThrough(*iConvSelu(2*k)),
            passThrough(*iConvSelu(2*k)),
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

@export
class iCNN3d(FlowNetwork):
    def __init__(self, in_channels=3, num_classes=10,res=32):
        super().__init__()
        self.num_classes = num_classes
        self.body = iSequential(
            iLogits(),
            *iConvSelu(in_channels),
            *iConvSelu(in_channels),
            *iConvSelu(in_channels),
            iAvgPool2d(),
            *iConvSelu(4*in_channels),
            *iConvSelu(4*in_channels),
            *iConvSelu(4*in_channels),
            iAvgPool2d(),
            *iConvSelu(16*in_channels),
            *iConvSelu(16*in_channels),
            *iConvSelu(16*in_channels),
            iAvgPool2d(),
            *iConvSelu(64*in_channels),
            *iConvSelu(64*in_channels),
            *iConvSelu(64*in_channels),
            iConv2d(64*in_channels,64*in_channels),
        )
        self.classifier_head = nn.Sequential(
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(64*in_channels,num_classes)
        )
        self.flow = iSequential(self.body,Flatten())
        self.prior = StandardNormal(in_channels*res*res)


@export
class iCNN3d2(FlowNetwork):
    def __init__(self, in_channels=3, num_classes=10,res=32):
        super().__init__()
        self.num_classes = num_classes
        self.body = nn.Sequential(
            conv2d(in_channels,in_channels),
            nn.ReLU(),
            conv2d(in_channels,in_channels),
            nn.ReLU(),
            conv2d(in_channels,in_channels),
            nn.ReLU(),
            NNdownsample(),
            conv2d(4*in_channels,4*in_channels),
            nn.ReLU(),
            conv2d(4*in_channels,4*in_channels),
            nn.ReLU(),
            conv2d(4*in_channels,4*in_channels),
            nn.ReLU(),
            NNdownsample(),
            conv2d(16*in_channels,16*in_channels),
            nn.ReLU(),
            conv2d(16*in_channels,16*in_channels),
            nn.ReLU(),
            conv2d(16*in_channels,16*in_channels),
            nn.ReLU(),
            NNdownsample(),
            conv2d(64*in_channels,64*in_channels),
            nn.ReLU(),
            conv2d(64*in_channels,64*in_channels),
            nn.ReLU(),
            conv2d(64*in_channels,64*in_channels),
            nn.ReLU(),
        )
        self.classifier_head = nn.Sequential(
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(64*in_channels,num_classes)
        )
        self.flow = iSequential(self.body,Flatten())
        self.prior = StandardNormal(in_channels*res*res)

@export
class iCNN3dCoords(FlowNetwork):
    def __init__(self, in_channels=3, num_classes=10,res=32):
        super().__init__()
        self.num_classes = num_classes
        self.body = iSequential(
            iLogits(),
            *[iCoordSelu(in_channels) for i in range(3)],
            iAvgPool2d(),
            *[iCoordSelu(4*in_channels) for i in range(3)],
            iAvgPool2d(),
            *[iCoordSelu(16*in_channels) for i in range(3)],
            iAvgPool2d(),
            *[iCoordSelu(64*in_channels) for i in range(3)],
            iConv2d(64*in_channels,64*in_channels),
        )
        self.classifier_head = nn.Sequential(
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(64*in_channels,num_classes)
        )
        self.flow = iSequential(self.body,Flatten())
        self.prior = StandardNormal(in_channels*res*res)

@export
class iLinear3d(iCNN3d):

    def __init__(self, num_classes=10,res=32):
        super().__init__()
        self.num_classes = num_classes
        self.body = iSequential(
            iLogits(),
            iCoordInjection(3),
            iConv2d(3,3),
            iConv2d(3,3),
            iConv2d(3,3),
            iAvgPool2d(),

            iCoordInjection(12),
            iConv2d(12,12),
            iConv2d(12,12),
            iConv2d(12,12),
            iAvgPool2d(),

            iCoordInjection(48),
            iConv2d(48,48),
            iConv2d(48,48),
            iConv2d(48,48),
            iAvgPool2d(),
            iCoordInjection(192),
            
            iConv2d(192,192),
            iConv2d(192,192),
            iConv2d(192,192),
        )
        self.classifier_head = nn.Sequential(
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(192,num_classes)
        )
        self.flow = iSequential(self.body,Flatten())
        self.prior = StandardNormal(3*res*res)
