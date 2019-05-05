import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from oil.utils.utils import Expression,export,Named
from oil.architectures.parts import ResBlock,conv2d
from blocks import ConcatResBlock,ODEBlock,RNNBlock
from blocks import ConcatBottleBlock,BezierResBlock
from downsample import SqueezeLayer,split,merge,padChannels,keepChannels
from clipped_BN import MeanOnlyBN, iBN
#from torch.nn.utils import spectral_norm
from auto_inverse import iSequential
from spectral_norm import SN


# def conv2d(in_channels,out_channels,kernel_size=3,dilation=1,**kwargs):
#     """ Wraps nn.Conv2d and CoordConv, padding is set to same
#         and coords=True can be specified to get additional coordinate in_channels"""
#     assert 'padding' not in kwargs, "assumed to be padding = same "
#     same = (kernel_size//2)*dilation
#     return nn.Conv2d(in_channels,out_channels,kernel_size,padding=0,dilation=dilation,**kwargs)


# def add_spectral_norm(module):
#     if isinstance(module,  (nn.ConvTranspose1d,
#                             nn.ConvTranspose2d,
#                             nn.ConvTranspose3d,
#                             nn.Conv1d,
#                             nn.Conv2d,
#                             nn.Conv3d)):
#         spectral_norm(module,dim = 1)
#         #print("SN on conv layer: ",module)
#     elif isinstance(module, nn.Linear):
#         spectral_norm(module,dim = 0)

class iResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,ksize=3,drop_rate=0,stride=1,
                    inverse_tol=1e-7,sigma=1.,**kwargs):
        super().__init__()
        assert stride==1, "Only stride 1 supported"
        #norm_layer = (lambda c: nn.GroupNorm(c//16,c)) if gn else nn.BatchNorm2d
        self.net = nn.Sequential(
            MeanOnlyBN(in_channels),
            nn.ReLU(),
            SN(conv2d(in_channels,out_channels,ksize,**kwargs)),
            MeanOnlyBN(out_channels),
            nn.ReLU(),
            SN(conv2d(out_channels,out_channels,ksize,**kwargs)),
            Expression(lambda x: sigma*x),
            nn.Dropout(p=drop_rate)
        )
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inverse_tol = inverse_tol
        self.reverse_iters = 0
        self.inverses_evaluated = 0
        self.x_y = None
        #self.apply(add_spectral_norm)

    @property
    def iters_per_reverse(self):
        return self.reverse_iters/self.inverses_evaluated

    def forward(self,inp):
        x,z = inp
        y = x + self.net(x)
        self.x_y = x,y
        return y,z # autograd will not traverse z_out?

    def inverse(self,output):
        y,z = output
        self.inverses_evaluated +=1
        with torch.no_grad():
            x_prev = y
            x_next = y
            diff = 1
            for i in range(100):#while diff > self.inverse_tol:
                x_next = y - self.net(x_prev)
                diff = torch.norm(x_next-x_prev)/(torch.max(torch.norm(y),torch.tensor(1e-15).to(y.device)))
                x_prev = x_next
                self.reverse_iters +=1
                if diff<self.inverse_tol: break
            #print(diff,self.reverse_iters)
        return x_prev,z

    def logdet(self):
        assert self.x_y is not None, "logdet called before forwards"
        x,y = self.x_y
        lndet = 0
        w = v = torch.randn_like(y)
        for k in range(1,6):
            w = jvp(x,y,w)
            #print(len(w))
            wTv = (w*v).sum(3).sum(2).sum(1) # dont sum over batch dim
            #print(f"iResblock mul {k}: {wTv}")
            lndet += ((-1)**(k+1))*wTv/k
        #print(w.shape)
        #print(f"iResblock logdet {lndet}")
        return lndet
            

def jvp(x,y,v,retain_graph=True):
    with torch.autograd.enable_grad():
        vJ = torch.autograd.grad(y,x,v,create_graph=retain_graph,retain_graph=retain_graph)[0]
    return vJ

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
    def logdet(self):
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
    def logdet(self):
        return 0

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

# Converts a list of torch.tensors into a single flat torch.tensor
def flatten(tensorList):
    flatList = []
    for t in tensorList:
        flatList.append(t.contiguous().view(t.numel()))
    return torch.cat(flatList)

# Takes a flat torch.tensor and unflattens it to a list of torch.tensors
#    shaped like likeTensorList
def unflatten_like(vector, likeTensorList):
    outList = []
    i=0
    for tensor in likeTensorList:
        n = tensor.numel()
        outList.append(vector[i:i+n].view(tensor.shape))
        i+=n
    return outList

@export
class iResnet(nn.Module,metaclass=Named):
    def __init__(self,num_classes=10,k=32,sigma=1.):
        super().__init__()
        self.num_classes = num_classes
        self.body = iSequential(
            padChannels(k-3),
            addZslot(),
            both(iBN(k),I),
            iResBlock(k,k,sigma=sigma),
            both(SqueezeLayer(2),I),
            both(iBN(4*k),I),
            iResBlock(4*k,4*k,sigma=sigma),
            keepChannels(2*k),
            both(iBN(2*k),I),
            iResBlock(2*k,2*k,sigma=sigma),
            both(SqueezeLayer(2),I),
            both(iBN(8*k),I),
            iResBlock(8*k,8*k,sigma=sigma),
            keepChannels(4*k),
            both(iBN(4*k),I),
            iResBlock(4*k,4*k,sigma=sigma),
            both(SqueezeLayer(2),I),
            both(iBN(16*k),I),
            iResBlock(16*k,16*k,sigma=sigma),
            keepChannels(8*k),
            both(iBN(8*k),I),
            iResBlock(8*k,8*k,sigma=sigma),
            Join(),
        )
        self.head = nn.Sequential(
            BNrelu(8*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(8*k,num_classes)
        )
        self.k = k
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

    @property
    def device(self):
        try: return self._device
        except AttributeError:
            self._device = next(self.parameters()).device
            return self._device

    def forward(self,x):
        return self.head(self.body(x)[-1])
    
    def get_all_z_squashed(self,x):
        return flatten(self.body(x))

    def logdet(self):
        return self.body.logdet()
    
    def inverse(self,z):
        return self.body.inverse(z)

    def sample(self,bs=1):
        z_all = [torch.randn(bs,*shape).to(self.device) for shape in self.z_shapes]
        return self.inverse(z_all)

    # def log_data(self,logger,step):
    #     for i,child in enumerate(self.body.named_children()):
    #         name,m = child
    #         if isinstance(m, SN):
    #             logger.add_scalars('info',{'Sigma_{}'.format(name):m._s.cpu().data},step)

@export
class iResnetLarge(iResnet):
    def __init__(self,num_classes=10,k=32,sigma=1.,block_size=4):
        super().__init__()
        self.num_classes = num_classes
        self.body = iSequential(
            padChannels(k-3),
            addZslot(),
            both(iBN(k),I),
            *[iResBlock(k,k,sigma=sigma) for _ in range(block_size)],
            both(SqueezeLayer(2),I),
            both(iBN(4*k),I),
            *[iResBlock(4*k,4*k,sigma=sigma) for _ in range(block_size)],
            both(SqueezeLayer(2),I),
            both(iBN(16*k),I),
            iResBlock(16*k,16*k,sigma=sigma),
            keepChannels(8*k),
            both(iBN(8*k),I),
            *[iResBlock(8*k,8*k,sigma=sigma) for _ in range(block_size)],
        )
        self.head = nn.Sequential(
            BNrelu(8*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(8*k,num_classes)
        )
@export
class iResnetLargeV2(nn.Module,metaclass=Named):
    def __init__(self,num_classes=10,k=64,sigma=1.,block_size=4):
        super().__init__()
        self.num_classes = num_classes
        self.foot = iSequential(
            padChannels(k-3),
            #iConv2d(k,k),
            addZslot(),
        )
        self.body = iSequential(
            both(iBN(k),I),
            *[iResBlock(k,k,sigma=sigma) for _ in range(block_size)],
            keepChannels(k//2),
            both(SqueezeLayer(2),I),
            both(iBN(2*k),I),
            *[iResBlock(2*k,2*k,sigma=sigma) for _ in range(block_size)],
            keepChannels(k),
            both(SqueezeLayer(2),I),
            both(iBN(4*k),I),
            *[iResBlock(4*k,4*k,sigma=sigma) for _ in range(block_size)],
            keepChannels(2*k),
            both(SqueezeLayer(2),I),
            both(iBN(8*k),I),
            *[iResBlock(8*k,8*k,sigma=sigma) for _ in range(block_size)],
            keepChannels(4*k),
            both(SqueezeLayer(2),I),
            both(iBN(16*k),I),
            *[iResBlock(16*k,16*k,sigma=sigma) for _ in range(block_size)],
        )
        self.head = nn.Sequential(
            BNrelu(16*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(16*k,num_classes)
        )
    def forward(self,x):
        y,z = self.body(self.foot(x))
        return self.head(y)
