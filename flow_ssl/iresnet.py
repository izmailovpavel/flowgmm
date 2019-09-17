import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from flow_ssl.utils import Expression,export,Named
from flow_ssl.invertible import SqueezeLayer,padChannels,keepChannels,NNdownsample,iAvgPool2d,RandomPadChannels,Flatten
#from torch.nn.utils import spectral_norm
from flow_ssl.invertible import iLogits, iBN, MeanOnlyBN, iSequential, passThrough, addZslot, Join, pad_circular_nd,SN
from flow_ssl.invertible import iConv2d, iResBlock
from flow_ssl.conv_parts import ResBlock,conv2d
from flow_ssl.icnn.icnn import FlowNetwork,StandardNormal
from flow_ssl.invertible import Swish, ActNorm1d, ActNorm2d
from flow_ssl.invertible.iresnet_trash import SpectralNormConv2d,SpectralNormLinear

def iResBlockConv(outer_channels,inner_channels):
    gnet = nn.Sequential(
        Swish(),
        SpectralNormConv2d(outer_channels,inner_channels,3,padding=1,atol=0.001,rtol=0.001,coeff=0.98,stride=1),
        Swish(),
        SpectralNormConv2d(inner_channels,inner_channels,1,padding=0,atol=0.001,rtol=0.001,coeff=0.98,stride=1),
        Swish(),
        SpectralNormConv2d(inner_channels,outer_channels,3,padding=1,atol=0.001,rtol=0.001,coeff=0.98,stride=1))
    return iSequential(iResBlock(gnet,n_dist='poisson'),ActNorm2d(outer_channels))

def iResBlockLinear(outer_channels,inner_channels):
    gnet = nn.Sequential(
        Swish(),
        SpectralNormLinear(outer_channels,inner_channels,atol=0.001,rtol=0.001,coeff=0.98),
        Swish(),
        SpectralNormLinear(inner_channels,inner_channels,atol=0.001,rtol=0.001,coeff=0.98),
        Swish(),
        SpectralNormLinear(inner_channels,outer_channels,atol=0.001,rtol=0.001,coeff=0.98))
    return iSequential(iResBlock(gnet,n_dist='poisson'),ActNorm1d(outer_channels))

@export
def SmallResidualFlow(in_channels):
    return ResidualFlow(in_channels,k=96,num_per_block=8)

@export
class ResidualFlow(FlowNetwork):
    def __init__(self, in_channels=3, num_classes=10, k=512,num_per_block=16):
        super().__init__()
        self.num_classes = num_classes
        self.flow = iSequential(
            #iLogits(),
            SqueezeLayer(),
            *[iResBlockConv(in_channels*4,k) for i in range(num_per_block)],
            SqueezeLayer(),
            *[iResBlockConv(in_channels*16,k) for i in range(num_per_block)],
            SqueezeLayer(),
            *[iResBlockConv(in_channels*64,k) for i in range(num_per_block)],
            Flatten(),
            *[iResBlockLinear(3*32*32,k//4) for i in range(4)],
        )
        self.k = k
        self.prior = lambda device: StandardNormal(3*32*32,device)

    def nll(self,x):
        x.requires_grad = True
        return super().nll(x)
    def forward(self,x):
        return self.nll(x)
# class iResBlock(nn.Module):
#     def __init__(self,in_channels,out_channels,ksize=3,drop_rate=0,stride=1,
#                     inverse_tol=1e-7,sigma=1.,**kwargs):
#         super().__init__()
#         assert stride==1, "Only stride 1 supported"
#         #norm_layer = (lambda c: nn.GroupNorm(c//16,c)) if gn else nn.BatchNorm2d
#         self.net = nn.Sequential(
#             MeanOnlyBN(in_channels),
#             nn.ReLU(),
#             SN(conv2d(in_channels,out_channels,ksize,**kwargs)),
#             MeanOnlyBN(out_channels),
#             nn.ReLU(),
#             SN(conv2d(out_channels,out_channels,ksize,**kwargs)),
#             Expression(lambda x: sigma*x),
#             nn.Dropout(p=drop_rate)
#         )
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.inverse_tol = inverse_tol
#         self.reverse_iters = 0
#         self.inverses_evaluated = 0
#         self.x_y = None
#         #self.apply(add_spectral_norm)

#     @property
#     def iters_per_reverse(self):
#         return self.reverse_iters/self.inverses_evaluated

#     def forward(self,x):
#         with torch.enable_grad():
#             if x.is_leaf: x.requires_grad=True
#             y = x + self.net(x)
#             self.x_y = x,y
#         return y

#     def inverse(self,y):
#         self.inverses_evaluated +=1
#         with torch.no_grad():
#             x_prev = y
#             x_next = y
#             diff = 1
#             for i in range(100):#while diff > self.inverse_tol:
#                 x_next = y - self.net(x_prev)
#                 diff = torch.norm(x_next-x_prev)/(torch.max(torch.norm(y),torch.tensor(1e-15).to(y.device)))
#                 x_prev = x_next
#                 self.reverse_iters +=1
#                 if diff<self.inverse_tol: break
#             #print(diff,self.reverse_iters)
#         return x_prev

#     def logdet(self):
#         assert self.x_y is not None, "logdet called before forwards"
#         x,y = self.x_y
#         lndet = 0
#         w = v = torch.randn_like(y)
#         for k in range(1,6):
#             w = jvp(x,y,w)
#             #print(len(w))
#             wTv = (w*v).sum(3).sum(2).sum(1) # dont sum over batch dim
#             #print(f"iResblock mul {k}: {wTv}")
#             lndet += ((-1)**(k+1))*wTv/k
#         #print(w.shape)
#         #print(f"iResblock logdet {lndet}")
#         return lndet

# class iBottleneck(iResBlock):
#     def __init__(self,in_channels,channels,ksize=3,drop_rate=0,stride=1,
#                     inverse_tol=1e-7,sigma=1.,**kwargs):
#         super().__init__(in_channels,channels,ksize,drop_rate,stride,
#                     inverse_tol,sigma,**kwargs)
#         self.net = nn.Sequential(
#             #MeanOnlyBN(in_channels),
#             nn.ELU(),
#             SN(conv2d(in_channels,channels,ksize,**kwargs)),
#             #MeanOnlyBN(channels),
#             nn.ELU(),
#             SN(conv2d(channels,channels,1,**kwargs)),
#             #MeanOnlyBN(channels),
#             nn.ELU(),
#             SN(conv2d(channels,in_channels,ksize,**kwargs)),
#             Expression(lambda x: sigma*x),
#             nn.Dropout(p=drop_rate)
#         )

# def jvp(x,y,v,retain_graph=True):
#     with torch.autograd.enable_grad():
#         vJ = torch.autograd.grad(y,x,v,create_graph=retain_graph,allow_unused=True)[0]
#     return vJ


# def BNrelu(channels,gn=False):
#     norm_layer = nn.GroupNorm(channels//16,channels) if gn else nn.BatchNorm2d(channels)
#     return nn.Sequential(nn.ReLU(),norm_layer)

# def BN(channels,gn=False):
#     return nn.GroupNorm(channels//16,channels) if gn else nn.BatchNorm2d(channels)



# class aResnet(nn.Module,metaclass=Named):
#     def __init__(self,num_classes=10,k=64,gn=False):
#         super().__init__()
#         self.num_classes = num_classes
#         self.net = nn.Sequential(
#             conv2d(3,k),
#             ResBlock(k,2*k,gn=gn,stride=2),
#             ResBlock(2*k,2*k,gn=gn),
#             ResBlock(2*k,4*k,gn=gn,stride=2),
#             ResBlock(4*k,4*k,gn=gn),
#             ResBlock(4*k,8*k,gn=gn,stride=2),
#             ResBlock(8*k,8*k,gn=gn),
#             BNrelu(8*k,gn=gn),
#             Expression(lambda u:u.mean(-1).mean(-1)),
#             nn.Linear(8*k,num_classes)
#         )
#     def forward(self,x):
#         return self.net(x)

# @export
# class iResnet(FlowNetwork):
#     def __init__(self,num_classes=10,k=32,sigma=.6):
#         super().__init__()
#         self.num_classes = num_classes
#         self.body = iSequential(
#             iLogits(),
#             RandomPadChannels(k-3),
#             addZslot(),
#             passThrough(iConv2d(k,k)),
#             passThrough(iBN(k)),
#             passThrough(iResBlock(k,k,sigma=sigma)),
#             passThrough(SqueezeLayer()),
#             passThrough(iBN(4*k)),
#             passThrough(iResBlock(4*k,4*k,sigma=sigma)),
#             keepChannels(2*k),

#             passThrough(iBN(2*k)),
#             passThrough(iResBlock(2*k,2*k,sigma=sigma)),
#             passThrough(SqueezeLayer()),
#             passThrough(iBN(8*k)),
#             passThrough(iResBlock(8*k,8*k,sigma=sigma)),
#             keepChannels(4*k),

#             passThrough(iBN(4*k)),
#             passThrough(iResBlock(4*k,4*k,sigma=sigma)),
#             passThrough(SqueezeLayer()),
#             passThrough(iBN(16*k)),
#             passThrough(iResBlock(16*k,16*k,sigma=sigma)),
#             keepChannels(8*k),
#             passThrough(iBN(8*k)),
#             passThrough(iResBlock(8*k,8*k,sigma=sigma)),
#             Join(),
#         )
#         self.classifier_head = nn.Sequential(
#             BNrelu(8*k),
#             Expression(lambda u:u.mean(-1).mean(-1)),
#             nn.Linear(8*k,num_classes)
#         )
#         self.k = k
#         self.flow = iSequential(self.body,Flatten())
#         self.prior = StandardNormal(k*32*32)

#     def nll(self,x):
#         x.requires_grad = True
#         return super().nll(x)
#     # def log_data(self,logger,step):
#     #     for i,child in enumerate(self.body.named_children()):
#     #         name,m = child
#     #         if isinstance(m, SN):
#     #             logger.add_scalars('info',{'Sigma_{}'.format(name):m._s.cpu().data},step)


# @export
# class iResnetProper(iResnet):
#     def __init__(self, in_channels=3, num_classes=10, k=256, sigma=.6):
#         super().__init__()
#         num_per_block = 5
#         self.num_classes = num_classes
#         self.body = iSequential(
#             iLogits(),
#             #iConv2d(3,3),

#             #iBN(3),
#             *[iBottleneck(in_channels,k//4,sigma=sigma) for i in range(num_per_block)],
#             NNdownsample(),
#             #iBN(12),
#             *[iBottleneck(in_channels*4,k//2,sigma=sigma) for i in range(num_per_block)],
#             NNdownsample(),
#             #iBN(48),
#             *[iBottleneck(in_channels*16,k,sigma=sigma) for i in range(num_per_block)],
#         )
#         self.classifier_head = nn.Sequential(
#             BNrelu(in_channels*16),
#             Expression(lambda u:u.mean(-1).mean(-1)),
#             nn.Linear(in_channels*16,num_classes)
#         )
#         self.k = k
#         self.flow = iSequential(self.body,Flatten())
#         self.prior = StandardNormal(3*32*32)

# class iResnetLarge(iResnet):
#     def __init__(self,num_classes=10,k=32,sigma=1.,block_size=4):
#         super().__init__()
#         self.num_classes = num_classes
#         self.body = iSequential(
#             padChannels(k-3),
#             addZslot(),
#             both(iBN(k),I),
#             *[iResBlock(k,k,sigma=sigma) for _ in range(block_size)],
#             both(SqueezeLayer(2),I),
#             both(iBN(4*k),I),
#             *[iResBlock(4*k,4*k,sigma=sigma) for _ in range(block_size)],
#             both(SqueezeLayer(2),I),
#             both(iBN(16*k),I),
#             iResBlock(16*k,16*k,sigma=sigma),
#             keepChannels(8*k),
#             both(iBN(8*k),I),
#             *[iResBlock(8*k,8*k,sigma=sigma) for _ in range(block_size)],
#         )
#         self.head = nn.Sequential(
#             BNrelu(8*k),
#             Expression(lambda u:u.mean(-1).mean(-1)),
#             nn.Linear(8*k,num_classes)
#         )

# class iResnetLargeV2(nn.Module,metaclass=Named):
#     def __init__(self,num_classes=10,k=64,sigma=1.,block_size=4):
#         super().__init__()
#         self.num_classes = num_classes
#         self.foot = iSequential(
#             padChannels(k-3),
#             #iConv2d(k,k),
#             addZslot(),
#         )
#         self.body = iSequential(
#             both(iBN(k),I),
#             *[iResBlock(k,k,sigma=sigma) for _ in range(block_size)],
#             keepChannels(k//2),
#             both(SqueezeLayer(2),I),
#             both(iBN(2*k),I),
#             *[iResBlock(2*k,2*k,sigma=sigma) for _ in range(block_size)],
#             keepChannels(k),
#             both(SqueezeLayer(2),I),
#             both(iBN(4*k),I),
#             *[iResBlock(4*k,4*k,sigma=sigma) for _ in range(block_size)],
#             keepChannels(2*k),
#             both(SqueezeLayer(2),I),
#             both(iBN(8*k),I),
#             *[iResBlock(8*k,8*k,sigma=sigma) for _ in range(block_size)],
#             keepChannels(4*k),
#             both(SqueezeLayer(2),I),
#             both(iBN(16*k),I),
#             *[iResBlock(16*k,16*k,sigma=sigma) for _ in range(block_size)],
#         )
#         self.head = nn.Sequential(
#             BNrelu(16*k),
#             Expression(lambda u:u.mean(-1).mean(-1)),
#             nn.Linear(16*k,num_classes)
#         )
#     def forward(self,x):
#         y,z = self.body(self.foot(x))
#         return self.head(y)
