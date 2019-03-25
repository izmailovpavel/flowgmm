import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from oil.utils.utils import Expression,export,Named
from oil.architectures.parts import conv2d,ResBlock
from blocks import ConcatResBlock,ODEBlock,RNNBlock
from blocks import ConcatBottleBlock,BezierResBlock
from downsample import SqueezeLayer,split,merge,padChannels



class iResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,ksize=3,drop_rate=0,stride=1,
                    gn=False,inverse_tol=1e-2,**kwargs):
        super().__init__()
        norm_layer = (lambda c: nn.GroupNorm(c//16,c)) if gn else nn.BatchNorm2d
        self.net = nn.Sequential(
            norm_layer(in_channels),
            nn.ReLU(),
            conv2d(in_channels,out_channels,ksize,**kwargs),
            norm_layer(out_channels),
            nn.ReLU(),
            conv2d(out_channels,out_channels,ksize,stride=stride,**kwargs),
            nn.Dropout(p=drop_rate)
        )
        assert in_channels >= out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inverse_tol = inverse_tol
        self.reverse_iters = 0
        self.inverses_evaluated = 0

    @property
    def iters_per_reverse(self):
        return self.reverse_iters/self.inverses_evaluated

    def forward(self,inp):
        x,z = inp
        shortcut = x
        if self.in_channels != self.out_channels:
            shortcut,z_extra = split(x,self.out_channels)
            z.append(z_extra)
        y = shortcut + self.net(x)
        return y,z # autograd will not traverse z_out?

    def inverse(self,output):
        y,z = output
        self.inverses_evaluated +=1
        if self.in_channels != self.out_channels: raise NotImplementedError
        with torch.no_grad():
            x_prev = y
            diff = 1
            while diff < self.inverse_tol:
                x_next = y - self.net(x_prev)
                diff = torch.norm(x_next-x_prev)/(torch.norm(x_prev)+1e-8)
                x_prev = x_next
                self.reverse_iters +=1
        return x_next,z

    def logdet(self,inp):
        raise NotImplementedError

def BNrelu(channels,gn=False):
    norm_layer = nn.GroupNorm(channels//16,channels) if gn else nn.BatchNorm2d(channels)
    return nn.Sequential(nn.ReLU(),norm_layer)

def BN(channels,gn=False):
    return nn.GroupNorm(channels//16,channels) if gn else nn.BatchNorm2d(channels)
    

@export
class aResnet(nn.Module,metaclass=Named):
    def __init__(self,num_classes=10,k=64,gn=False):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            conv2d(3,k),
            ResBlock(k,2*k,gn=gn,stride=2),
            ResBlock(2*k,2*k,gn=gn),
            ResBlock(2*k,4*k,gn=gn,stride=2),
            ResBlock(4*k,4*k,gn=gn),
            ResBlock(4*k,8*k,gn=gn,stride=2),
            ResBlock(8*k,8*k,gn=gn),
            BNrelu(8*k,gn=gn),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(8*k,num_classes)
        )
    def forward(self,x):
        return self.net(x)

@export
class squeezeResnet(nn.Module,metaclass=Named):
    def __init__(self,num_classes=10,k=64,gn=False):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            conv2d(3,k),
            ResBlock(k,k,gn=gn),
            SqueezeLayer(2),
            ResBlock(4*k,2*k,gn=gn),
            ResBlock(2*k,2*k,gn=gn),
            SqueezeLayer(2),
            ResBlock(8*k,4*k,gn=gn),
            ResBlock(4*k,4*k,gn=gn),
            SqueezeLayer(2),
            ResBlock(16*k,8*k,gn=gn),
            ResBlock(8*k,8*k,gn=gn),
            BNrelu(8*k,gn=gn),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(8*k,num_classes)
        )
    def forward(self,x):
        return self.net(x)

# Id = nn.Sequential

# class both(nn.Module):
#     def __init__(self,module1,module2):
#         super().__init__()
#         self.module1 = module1
#         self.module2 = module2
#     def forward(self,x,z):
#         return self.module1(x),self.module2(z)
#     def inverse(self,y,z_out):
#         return self.module1.inverse(y),self.module2.inverse(z_out)

@export
class iResnet(nn.Module,metaclass=Named):
    def __init__(self,num_classes=10,k=64,gn=False):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            padChannels(k-3),
            conv2d(k,k),
            Expression(lambda x: (x,[])),
            iResBlock(k,k,gn=gn),
            SqueezeLayer(2),
            iResBlock(4*k,2*k,gn=gn),
            iResBlock(2*k,2*k,gn=gn),
            SqueezeLayer(2),
            iResBlock(8*k,4*k,gn=gn),
            iResBlock(4*k,4*k,gn=gn),
            SqueezeLayer(2),
            iResBlock(16*k,8*k,gn=gn),
            iResBlock(8*k,8*k,gn=gn),
        )
        self.head = nn.Sequential(
            BNrelu(8*k,gn=gn),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(8*k,num_classes)
        )
    def forward(self,x):
        y,z = self.net(x)
        return self.head(y)