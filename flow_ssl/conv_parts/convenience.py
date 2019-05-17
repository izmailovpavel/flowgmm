import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_ssl.utils import export,Expression
#from ..invertible import iConv2d
try: 
    from oil.architectures.parts.CoordConv import CoordConv
except:
    def CoordConv(*args,**kwargs):
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


@export
def ConvBNrelu(in_channels,out_channels,stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1,stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

#@export
#def CircBNrelu(in_channels,out_channels):
#    return nn.Sequential(
#        iConv2d(in_channels,out_channels),
#        nn.BatchNorm2d(out_channels),
#        nn.ReLU()
#    )

@export
class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,ksize=3,drop_rate=0,stride=1,gn=False,**kwargs):
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
        if in_channels != out_channels:
            self.shortcut = conv2d(in_channels,out_channels,1,stride=stride,**kwargs)
        elif stride!=1:
            self.shortcut = Expression(lambda x: F.interpolate(x,scale_factor=1/stride))
        else:
            self.shortcut = nn.Sequential()

    def forward(self,x):
        return self.shortcut(x) + self.net(x)
