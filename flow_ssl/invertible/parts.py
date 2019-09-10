import torch
import torch.nn as nn
import numpy as np
from ..utils import export
from torch.nn import functional as F

@export
class iSequential(torch.nn.Sequential):

    def inverse(self,y):
        for module in reversed(self._modules.values()):
            assert hasattr(module,'inverse'), '{} has no inverse defined'.format(module)
            y = module.inverse(y)
        return y

    def logdet(self):
        log_det = 0
        for module in self._modules.values():
            assert hasattr(module,'logdet'), '{} has no logdet defined'.format(module)
            log_det += module.logdet()
        return log_det

    def reduce_func_singular_values(self,func):
        val = 0
        for module in self._modules.values():
            if hasattr(module,'reduce_func_singular_values'):
                val += module.reduce_func_singular_values(func)
        return val

@export
class addZslot(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x,[]
    def inverse(self,output):
        x,z = output
        assert not z, "nonempty z received"
        return x
    def logdet(self):
        return 0

@export
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,z):
        # If just a single tensor, early exit with a view
        if isinstance(z,torch.Tensor):
            self._shapes = z.shape[1:]
            return z.view(z.shape[0],-1)
        bs = z[-1].shape[0]
        self._shapes = [zi.shape[1:] for zi in z]
        return torch.cat([zi.view(bs,-1) for zi in z],dim=1)

    def inverse(self,z_flat):
        bs = z_flat.shape[0] # Early exit for single tensor
        if isinstance(self._shapes,torch.Size): return z_flat.view((bs,*self._shapes))
        dimensions = [np.prod(shape) for shape in self._shapes]
        z = [flat_part.view((bs, *shape)) for (flat_part, shape) in 
             zip(torch.split(z_flat,dimensions,dim=1),self._shapes)]
        return z
    def logdet(self):
        return 0


@export
class Join(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        y,z = x
        z.append(y)
        return z
    def inverse(self,z):
        z,y = z[:-1],z[-1]
        return y,z
    def logdet(self):
        return 0

@export
def FlatJoin():
    return iSequential(Join(),Flatten())

@export
class Id(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x
    def inverse(self,y):
        return y
    def logdet(self):
        return 0

I = Id()

@export
class both(nn.Module):
    def __init__(self,module1,module2):
        super().__init__()
        self.module1 = module1
        self.module2 = module2
    def forward(self,inp):
        x,z = inp
        return self.module1(x),self.module2(z)
    def inverse(self,output):
        y,z_out = output
        return self.module1.inverse(y),self.module2.inverse(z_out)
    def logdet(self):
        return self.module1.logdet() + self.module2.logdet()
    def reduce_func_singular_values(self,func):
        out = 0
        if hasattr(self.module1,'reduce_func_singular_values'):
            out +=self.module1.reduce_func_singular_values(func)
        if hasattr(self.module2,'reduce_func_singular_values'):
            out +=self.module2.reduce_func_singular_values(func) 
        return out


@export
def passThrough(*layers):
    return iSequential(*[both(layer,I) for layer in layers])


@export
def ActNorm(num_channels):
    return iSequential(ActNormShift(num_channels), ActNormScale(num_channels))


class ActNormShift(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.register_parameter("shift", 
                torch.nn.Parameter(torch.zeros([1, num_channels, 1, 1])))
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            self.shift.data = -x.mean(dim=(0, 2, 3)).view_as(self.shift)
            self.initialized = True
        self._logdet = 0.
        #print("ActNorm shift:", (x + self.shift).mean(dim=(0, 2, 3)))
        return x + self.shift 

    def inverse(self, x):
        return x - self.shift 

    def logdet(self):
        return self._logdet


class ActNormScale(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.register_parameter("log_scale", 
                torch.nn.Parameter(torch.zeros([1, num_channels, 1, 1])))
        self.initialized = False

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def forward(self, x):
        if not self.initialized:
            x_var = (x**2).mean(dim=(0, 2, 3)).view_as(self.scale)
            self.log_scale.data = -torch.log(x_var + 1e-6) / 2
            self.initialized = True
        self._logdet = x.shape[2] * x.shape[3] * self.log_scale.sum().expand(x.shape[0])
        #print("ActNorm scale:", ((x * self.scale)**2).mean(dim=(0, 2, 3)))
        return x * self.scale

    def inverse(self, x):
        return x / self.scale 

    def logdet(self):
        return self._logdet
    