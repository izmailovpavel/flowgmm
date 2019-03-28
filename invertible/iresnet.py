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
from downsample import SqueezeLayer,split,merge,padChannels,keepChannels
from clipped_BN import clippedBN
from torch.nn.utils import spectral_norm
import auto_inverse

def add_spectral_norm(module):
    if isinstance(module,  (nn.ConvTranspose1d,
                            nn.ConvTranspose2d,
                            nn.ConvTranspose3d,
                            nn.Conv1d,
                            nn.Conv2d,
                            nn.Conv3d)):
        spectral_norm(module,dim = 1)
        #print("SN on conv layer: ",module)
    elif isinstance(module, nn.Linear):
        spectral_norm(module,dim = 0)

class iResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,ksize=3,drop_rate=0,stride=1,
                    gn=False,inverse_tol=1e-2,**kwargs):
        super().__init__()
        #norm_layer = (lambda c: nn.GroupNorm(c//16,c)) if gn else nn.BatchNorm2d
        self.net = nn.Sequential(
            clippedBN(in_channels),
            nn.ReLU(),
            conv2d(in_channels,out_channels,ksize,**kwargs),
            clippedBN(out_channels),
            nn.ReLU(),
            conv2d(out_channels,out_channels,ksize,stride=stride,**kwargs),
            nn.Dropout(p=drop_rate)
        )
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inverse_tol = inverse_tol
        self.reverse_iters = 0
        self.inverses_evaluated = 0
        self.apply(add_spectral_norm)

    @property
    def iters_per_reverse(self):
        return self.reverse_iters/self.inverses_evaluated

    def forward(self,inp):
        x,z = inp
        y = x + self.net(x)
        return y,z # autograd will not traverse z_out?

    def inverse(self,output):
        y,z = output
        self.inverses_evaluated +=1
        with torch.no_grad():
            x_prev = y
            diff = 1
            while diff < self.inverse_tol:
                x_next = y - self.net(x_prev)
                diff = torch.norm(x_next-x_prev)/(torch.norm(x_prev)+1e-8)
                x_prev = x_next
                self.reverse_iters +=1
        return x_prev,z

    def logdet(self,inp):
        raise NotImplementedError

@export
def conv2d(in_channels,out_channels,kernel_size=3,coords=False,dilation=1,**kwargs):
    """ Wraps nn.Conv2d and CoordConv, padding is set to same
        and coords=True can be specified to get additional coordinate in_channels"""
    assert 'padding' not in kwargs, "assumed to be padding = same "
    same = (kernel_size//2)*dilation
    if coords: 
        return CoordConv(in_channels,out_channels,kernel_size,padding=same,dilation=dilation,**kwargs)
    else: 
        return nn.Conv2d(in_channels,out_channels,kernel_size,padding=same,dilation=dilation,**kwargs)

class iConv2d(nn.Module):
    """ wraps conv2d in a module with an inverse function """
    def __init__(self,*args,inverse_tol=1e-2,**kwargs):
        super().__init__()
        self.conv = conv2d(*args,**kwargs)
        self.inverse_tol = inverse_tol
        self.inverse_tol = inverse_tol
        self.reverse_iters = 0
        self.inverses_evaluated = 0
    def forward(self,inp):
        return self.conv(inp)
    def inverse(self,output):
        pass
    def logdet(self,inp):
        raise NotImplementedError

class addZslot(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x,[]
    def inverse(self,output):
        x,z = output
        assert not z, "nonempty z received"
        return x
    def logdet(self,inp):
        return 0

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

# @export
# class squeezeResnet(nn.Module,metaclass=Named):
#     def __init__(self,num_classes=10,k=64,gn=False):
#         super().__init__()
#         self.num_classes = num_classes
#         self.net = nn.Sequential(
#             conv2d(3,k),
#             ResBlock(k,k,gn=gn),
#             SqueezeLayer(2),
#             ResBlock(4*k,2*k,gn=gn),
#             ResBlock(2*k,2*k,gn=gn),
#             SqueezeLayer(2),
#             ResBlock(8*k,4*k,gn=gn),
#             ResBlock(4*k,4*k,gn=gn),
#             SqueezeLayer(2),
#             ResBlock(16*k,8*k,gn=gn),
#             ResBlock(8*k,8*k,gn=gn),
#             BNrelu(8*k,gn=gn),
#             Expression(lambda u:u.mean(-1).mean(-1)),
#             nn.Linear(8*k,num_classes)
#         )
#     def forward(self,x):
#         return self.net(x)

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
        self.foot = nn.Sequential(
            padChannels(k-3),
            conv2d(k,k),
            addZslot(),
        )
        self.body = nn.Sequential(
            iResBlock(k,k,gn=gn),
            SqueezeLayer(2),
            iResBlock(4*k,4*k,gn=gn),
            keepChannels(2*k),
            iResBlock(2*k,2*k,gn=gn),
            SqueezeLayer(2),
            iResBlock(8*k,8*k,gn=gn),
            keepChannels(4*k),
            iResBlock(4*k,4*k,gn=gn),
            SqueezeLayer(2),
            iResBlock(16*k,16*k,gn=gn),
            keepChannels(8*k),
            iResBlock(8*k,8*k,gn=gn),
        )
        self.head = nn.Sequential(
            BNrelu(8*k,gn=gn),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(8*k,num_classes)
        )
    def forward(self,x):
        y,z = self.body(self.foot(x))
        return self.head(y)