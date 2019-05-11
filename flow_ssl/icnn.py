import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from oil.utils.utils import Expression,export,Named
from oil.architectures.parts import ResBlock,conv2d
from .downsample import SqueezeLayer,split,merge,padChannels,keepChannels,NNdownsample,iAvgPool2d
from .downsample import iLogits
from .clipped_BN import MeanOnlyBN, iBN
#from torch.nn.utils import spectral_norm
from .auto_inverse import iSequential
import scipy as sp
import scipy.sparse
from .iresnet import both, I, addZslot, Join,flatten
from .spectral_norm import pad_circular_nd


class iEluNet(nn.Module,metaclass=Named):
    """
    Very small CNN
    """
    def __init__(self, num_classes=10,k=16):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.body = iSequential(
            padChannels(k-3),
            *iConvBNelu(k),
            *iConvBNelu(k),
            *iConvBNelu(k),
            NNdownsample(),#SqueezeLayer(2),
            #Expression(lambda x: torch.cat((x[:,:k],x[:,3*k:]),dim=1)),
            *iConvBNelu(4*k),
            *iConvBNelu(4*k),
            *iConvBNelu(4*k),
            NNdownsample(),#SqueezeLayer(2),
            #Expression(lambda x: torch.cat((x[:,:2*k],x[:,6*k:]),dim=1)),
            *iConvBNelu(16*k),
            *iConvBNelu(16*k),
            *iConvBNelu(16*k),
            iConv2d(16*k,16*k),
        )
        self.head = nn.Sequential(
            nn.BatchNorm2d(16*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(16*k,num_classes)
        )
    def forward(self,x):
        z = self.body(x)
        return self.head(z)
    def logdet(self):
        return self.body.logdet()
    def get_all_z_squashed(self,x):
        return self.body(x).reshape(-1)
    def inverse(self,z):
        return self.body.inverse(z)
    # def sample(self,bs=1):
    #     z = torch.randn(bs,16*self.k,32//4,32//4).to(self.device)
    #     return self.inverse(z)
    @property
    def device(self):
        try: return self._device
        except AttributeError:
            self._device = next(self.parameters()).device
            return self._device

    def prior_nll(self,z):
        d = z.shape[1]
        return .5*(z*z).sum(-1) + .5*np.log(2*np.pi)*d

    def nll(self,x):
        z = self.get_all_z_squashed(x).reshape(x.shape[0],-1)
        logdet = self.logdet()
        return  self.prior_nll(z) - logdet


class iEluNetMultiScale(iEluNet):
    def __init__(self,num_classes=10,k=16):
        super().__init__()
        self.num_classes = num_classes
        self.body = iSequential(

        )
    def __init__(self, num_classes=10,k=32):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.body = iSequential(
            padChannels(k-3),
            addZslot(),

            passThrough(*iConvBNelu(k)),
            passThrough(*iConvBNelu(k)),
            passThrough(*iConvBNelu(k)),
            passThrough(NNdownsample()),#SqueezeLayer(2)),
            passThrough(iConv1x1(4*k)),
            keepChannels(2*k),
            
            passThrough(*iConvBNelu(2*k)),
            passThrough(*iConvBNelu(2*k)),
            passThrough(*iConvBNelu(2*k)),
            passThrough(NNdownsample()),
            passThrough(iConv1x1(8*k)),# (replace with iConv1x1 or glow style 1x1)
            keepChannels(4*k),
            
            passThrough(*iConvBNelu(4*k)),
            passThrough(*iConvBNelu(4*k)),
            passThrough(*iConvBNelu(4*k)),
            passThrough(iConv2d(4*k,4*k)),
            Join(),
        )
        self.head = nn.Sequential(
            nn.BatchNorm2d(4*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(4*k,num_classes)
        )
    @property
    def z_shapes(self):
        # For CIFAR10: starting size = 32x32
        h = w = 32
        channels = self.k
        shapes = []
        for module in self.body:
            if isinstance(module,keepChannels):
                #print(module)
                channels = 2*channels
                h //=2
                w //=2
                shapes.append((channels,h,w))
        shapes.append((channels,h,w))
        return shapes

    def get_all_z_squashed(self,x):
        return flatten(self.body(x))

    def forward(self,x):
        z = self.body(x)
        return self.head(z[-1])
    def sample(self,bs=1):
        z_all = [torch.randn(bs,*shape).to(self.device) for shape in self.z_shapes]
        return self.inverse(z_all)

class iEluNetMultiScaleLarger(iEluNetMultiScale):
    def __init__(self, num_classes=10,k=128):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.body = iSequential(
            padChannels(k-3),
            addZslot(),
            passThrough(*iConvBNelu(k)),
            passThrough(*iConvBNelu(k)),
            passThrough(*iConvBNelu(k)),
            passThrough(NNdownsample()),
            #passThrough(iConv1x1(4*k)),
            keepChannels(2*k),
            passThrough(*iConvBNelu(2*k)),
            passThrough(*iConvBNelu(2*k)),
            passThrough(*iConvBNelu(2*k)),
            passThrough(NNdownsample()),
            #passThrough(iConv1x1(8*k)),
            keepChannels(2*k),
            passThrough(*iConvBNelu(2*k)),
            passThrough(*iConvBNelu(2*k)),
            passThrough(*iConvBNelu(2*k)),
            #passThrough(iConv2d(2*k,2*k)),
            Join(),
        )
        self.head = nn.Sequential(
            #nn.BatchNorm2d(2*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(2*k,num_classes)
        )
    @property
    def z_shapes(self):
        # For CIFAR10: starting size = 32x32
        h = w = 32
        k = self.k
        shapes = [(2*k,h//2,w//2),(6*k,h//4,w//4),(2*k,h//4,w//4)]
        return shapes


class DegredationTester(nn.Module):
    def __init__(self, num_classes=10,k=128,circ=False,slrelu=False,lrelu=None,ds='max'):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        
        conv = lambda c1,c2: iConv2d(c1,c2,circ=circ)
        BN = nn.BatchNorm2d
        relu = iSLReLU if slrelu else nn.ReLU
        if lrelu is not None: relu = lambda: nn.LeakyReLU(lrelu)
        if ds=='max': downsample = nn.MaxPool2d(2)
        elif ds=='checkerboard': downsample = SqueezeLayer(2)
        elif ds=='nn': downsample = NNdownsample()
        elif ds=='avg': downsample = iAvgPool2d()
        else: assert False, "unknown option"
        CBR = lambda c1,c2: nn.Sequential(conv(c1,c2),BN(c2),relu())
        self.net = nn.Sequential(
            CBR(3,k),
            CBR(k,k),
            CBR(k,2*k),
            downsample,
            Expression(lambda x: x[:,:2*k]),
            CBR(2*k,2*k),
            CBR(2*k,2*k),
            CBR(2*k,2*k),
            downsample,
            Expression(lambda x: x[:,:2*k]),
            CBR(2*k,2*k),
            CBR(2*k,2*k),
            CBR(2*k,2*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(2*k,num_classes)
        )
    def forward(self,x):
        return self.net(x)



class iEluNet3d(iEluNetMultiScale):
    def __init__(self, num_classes=10,k=64):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.body = iSequential(
            iLogits(),
            *iConvBNelu(3),
            *iConvBNelu(3),
            *iConvBNelu(3),
            NNdownsample(),
            *iConvBNelu(12),
            *iConvBNelu(12),
            *iConvBNelu(12),
            NNdownsample(),
            *iConvBNelu(48),
            *iConvBNelu(48),
            *iConvBNelu(48),
            NNdownsample(),
            *iConvBNelu(192),
            *iConvBNelu(192),
            *iConvBNelu(192),
            iConv2d(192,192),
        )
        self.head = nn.Sequential(
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(192,num_classes)
        )

    def sample(self,bs=1):
        z_all = torch.randn(bs,192,32//8,32//8).to(self.device)
        return self.inverse(z_all)
    @property
    def z_shapes(self):
        # For CIFAR10: starting size = 32x32
        h = w = 32
        k = self.k
        shapes = [(192,h//8,h//8)]#[(3*2**6-2*k,h//8,w//8),(2*k,h//8,w//8)]#[(48,h//4,h//4)]#
        return shapes



class iLinear(iEluNet3d):
    def __init__(self, num_classes=10,k=128):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.body = iSequential(
            iLogits(),
            iConv2d(3,3),
            iConv2d(3,3),
            iConv2d(3,3),
            NNdownsample(),
            iConv2d(12,12),
            iConv2d(12,12),
            iConv2d(12,12),
            NNdownsample(),
            iConv2d(48,48),
            iConv2d(48,48),
            iConv2d(48,48),
            NNdownsample(),
            iConv2d(192,192),
            iConv2d(192,192),
            iConv2d(192,192),
        )
        self.head = nn.Sequential(
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(192,num_classes)
        )

    def sample(self,bs=1):
        z_all = torch.randn(bs,192,32//8,32//8).to(self.device)
        return self.inverse(z_all)
    @property
    def z_shapes(self):
        # For CIFAR10: starting size = 32x32
        h = w = 32
        k = self.k
        shapes = [(192,h//8,h//8)]#[(3*2**6-2*k,h//8,w//8),(2*k,h//8,w//8)]#[(48,h//4,h//4)]#
        return shapes
