import torch
import torch.nn as nn
import numpy as np
from ..utils import export

@export
class iSequential(torch.nn.Sequential):

    def inverse(self,y):
        j = len(self._modules.values())
        for module in reversed(self._modules.values()):
            j -=1
            #print(f"Inverting layer{j} with module {module}")
            assert hasattr(module,'inverse'), '{} has no inverse defined'.format(module)
            y = module.inverse(y)
        return y

    def logdet(self):
        log_det = 0
        for module in self._modules.values():
            #print(module)
            assert hasattr(module,'logdet'), '{} has no logdet defined'.format(module)
            log_det += module.logdet()
        return log_det

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

# Converts a list of torch.tensors into a single flat torch.tensor
def flatten(tensorList):
    flatList = []
    for t in tensorList:
        flatList.append(t.contiguous().view(t.numel()))
    return torch.cat(flatList)



# # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
# #    shaped like likeTensorList
# def unflatten_like(vector, likeTensorList):
#     outList = []
#     i=0
#     for tensor in likeTensorList:
#         n = tensor.numel()
#         outList.append(vector[i:i+n].view(tensor.shape))
#         i+=n
#     return outList


@export
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,z):
        bs = z[-1].shape[0]
        self._shapes = [zi.shape[1:] for zi in z]
        return torch.cat([zi.view(bs,-1) for zi in z],dim=1)

    def inverse(self,z_flat):
        bs = z_flat.shape[0]
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
    return iSequential(Join,Flatten)

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


@export
def passThrough(*layers):
    return iSequential(*[both(layer,I) for layer in layers])
