import torch
import torch.nn as nn
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
@export
class Join(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        y,z = x
        z.append(y)
        return z
    def inverse(self,z):
        #y = z.pop()
        z,y = z[:-1],z[-1]
        return y,z
    def logdet(self):
        return 0

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


# def iConvBNelu(ch):
#     return iSequential(iConv2d(ch,ch),iSLReLU(.1))#iSequential(iConv2d(ch,ch),iBN(ch),iSLReLU())

# def ConvBNrelu(in_channels,out_channels,stride=1):
#     return nn.Sequential(
#         nn.Conv2d(in_channels,out_channels,3,padding=1,stride=stride),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU()
#     )

# def CircBNrelu(in_channels,out_channels):
#     return nn.Sequential(
#         iConv2d(in_channels,out_channels),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU()
#     )
