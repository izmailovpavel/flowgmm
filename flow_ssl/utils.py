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

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)