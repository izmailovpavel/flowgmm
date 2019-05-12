import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from oil.utils.utils import Expression,export,Named
from ..utils import conv2d,ResBlock

from .blocks import ConcatResBlock,ODEBlock,RNNBlock
from .blocks import ConcatBottleBlock,BezierResBlock

def BNrelu(channels,gn=False):
    norm_layer = nn.GroupNorm(channels//16,channels) if gn else nn.BatchNorm2d(channels)
    return nn.Sequential(nn.ReLU(),norm_layer)


class SmallResnet(nn.Module,metaclass=Named):
    def __init__(self,num_classes=10,k=64,gn=False,block_size=4):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            conv2d(3,k),
            *[ResBlock(k,k,gn=gn) for i in range(block_size)],
            ResBlock(k,2*k,gn=gn,stride=2),
            *[ResBlock(2*k,2*k,gn=gn) for i in range(block_size)],
            ResBlock(2*k,4*k,gn=gn,stride=2),
            *[ResBlock(4*k,4*k,gn=gn) for i in range(block_size)],
            BNrelu(4*k,gn=gn),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(4*k,num_classes)
        )
    def forward(self,x):
        return self.net(x)


class LongResnet(nn.Module,metaclass=Named):
    def __init__(self,num_classes=10,k=64,gn=False,block_size=12):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            conv2d(3,k),
            ResBlock(k,2*k,gn=gn,stride=2),
            ResBlock(2*k,4*k,gn=gn,stride=2),
            *[ResBlock(4*k,4*k,gn=gn) for i in range(block_size)],
            BNrelu(4*k,gn=gn),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(4*k,num_classes)
        )
    def forward(self,x):
        return self.net(x)


class ODEResnet(nn.Module,metaclass=Named):
    def __init__(self,num_classes=10,k=64,gn=False):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            conv2d(3,k),
            ResBlock(k,2*k,gn=gn,stride=2),
            ResBlock(2*k,4*k,gn=gn,stride=2),
            ODEBlock(ConcatResBlock(4*k,gn=gn)),
            BNrelu(4*k,gn=gn),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(4*k,num_classes)
        )
    def forward(self,x):
        return self.net(x)
    def log_data(self,logger,step):
        for name, m in self.net.named_children():
            if isinstance(m, ODEBlock):
                logger.add_scalars('info',{'nfe{}'.format(name):m.nfe},step)


class BezierODE(nn.Module,metaclass=Named):
    def __init__(self,num_classes=10,k=64,gn=False,block_size=12):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            conv2d(3,k),
            ResBlock(k,2*k,gn=gn,stride=2),
            ResBlock(2*k,4*k,gn=gn,stride=2),
            ODEBlock(BezierResBlock(4*k,gn=gn,add=True,bends=block_size),tol=1e-1),
            BNrelu(4*k,gn=gn),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(4*k,num_classes)
        )
    def forward(self,x):
        return self.net(x)

    def log_data(self,logger,step):
        for name, m in self.net.named_children():
            if isinstance(m, ODEBlock):
                logger.add_scalars('info',{'nfe{}'.format(name):m.nfe},step)



class SplitODEResnet(nn.Module,metaclass=Named):
    def __init__(self,num_classes=10,k=64,gn=False,block_size=4):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            conv2d(3,k),
            ODEBlock(ConcatResBlock(k,gn=gn)),
            ResBlock(k,2*k,gn=gn,stride=2),
            ODEBlock(ConcatResBlock(2*k,gn=gn)),
            ResBlock(2*k,4*k,gn=gn,stride=2),
            ODEBlock(ConcatResBlock(4*k,gn=gn)),
            BNrelu(4*k,gn=gn),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(4*k,num_classes)
        )
    def forward(self,x):
        return self.net(x)
    def log_data(self,logger,step):
        for name, m in self.net.named_children():
            if isinstance(m, ODEBlock):
                logger.add_scalars('info',{'nfe{}'.format(name):m.nfe},step)


class RNNResnet(nn.Module,metaclass=Named):
    def __init__(self,num_classes=10,k=64,gn=False,block_size=12):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            conv2d(3,k),
            ResBlock(k,2*k,gn=gn,stride=2),
            ResBlock(2*k,4*k,gn=gn,stride=2),
            RNNBlock(ConcatResBlock(4*k,gn=gn,add=True),L=block_size),
            BNrelu(4*k,gn=gn),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(4*k,num_classes)
        )
    def forward(self,x):
        return self.net(x)


class RNNBottle(nn.Module,metaclass=Named):
    def __init__(self,num_classes=10,k=64,gn=False,block_size=12):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            conv2d(3,k),
            ResBlock(k,2*k,gn=gn,stride=2),
            ResBlock(2*k,4*k,gn=gn,stride=2),
            BNrelu(4*k,gn=gn),conv2d(4*k,32*k,1),
            RNNBlock(ConcatBottleBlock(32*k,gn=gn,add=True),L=block_size),
            BNrelu(32*k,gn=gn),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(32*k,num_classes)
        )
    def forward(self,x):
        return self.net(x)

class BezierRNN(nn.Module,metaclass=Named):
    def __init__(self,num_classes=10,k=64,gn=False,block_size=12):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            conv2d(3,k),
            ResBlock(k,2*k,gn=gn,stride=2),
            ResBlock(2*k,4*k,gn=gn,stride=2),
            RNNBlock(BezierResBlock(4*k,gn=gn,add=True,bends=block_size//2),L=block_size),
            BNrelu(4*k,gn=gn),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(4*k,num_classes)
        )
    def forward(self,x):
        return self.net(x)

class BezierRNNSplit(nn.Module,metaclass=Named):
    def __init__(self,num_classes=10,k=64,gn=False,block_size=4):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            conv2d(3,k),
            RNNBlock(BezierResBlock(k,gn=gn,add=True,bends=block_size),L=block_size*2),
            ResBlock(k,2*k,gn=gn,stride=2),
            RNNBlock(BezierResBlock(2*k,gn=gn,add=True,bends=block_size),L=block_size*2),
            ResBlock(2*k,4*k,gn=gn,stride=2),
            RNNBlock(BezierResBlock(4*k,gn=gn,add=True,bends=block_size),L=block_size*2),
            BNrelu(4*k,gn=gn),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(4*k,num_classes)
        )
    def forward(self,x):
        return self.net(x)