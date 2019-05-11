import sys
import torch
import torch.nn as nn

class Named(type):
    def __str__(self):
        return self.__name__

def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn

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