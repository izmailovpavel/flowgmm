import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from oil.utils.utils import Expression,export,Named
from oil.architectures.parts import ResBlock,conv2d
from downsample import SqueezeLayer,split,merge,padChannels,keepChannels
from clipped_BN import MeanOnlyBN, iBN
#from torch.nn.utils import spectral_norm
from auto_inverse import iSequential
import scipy as sp
import scipy.sparse
from iresnet import both, I, addZslot, Join,flatten
class iConv2d(nn.Module):
    """ wraps conv2d in a module with an inverse function """
    def __init__(self,*args,inverse_tol=1e-7,**kwargs):
        super().__init__()
        self.conv = conv2d(*args,**kwargs)
        self.inverse_tol = inverse_tol
        self._reverse_iters = 0
        self._inverses_evaluated = 0
    @property
    def iters_per_reverse(self):
        return self._reverse_iters/self._inverses_evaluated
    def forward(self,x):
        self._shape = x.shape
        return self.conv(x)
    # GMRES inverse method
    # def inverse(self,y):
    #     self._inverses_evaluated +=1
    #     d = np.prod(self._shape[1:])
    #     A = sp.sparse.linalg.LinearOperator((d,d),matvec=self.np_matvec)
    #     bs = y.shape[0]
    #     x = torch.zeros_like(y)
    #     conv_diag = torch.diag(self.conv.weight[:,:,1,1]).cpu().data.numpy()[:,None,None] #shape c x 1 x 1
    #     inv_diag_operator = sp.sparse.linalg.LinearOperator((d,d),
    #         matvec=lambda v: (v.reshape(*self._shape[1:]).astype(np.float32)/conv_diag).reshape(-1))
    #     for i in range(bs):
    #         np_y = (y[i]-self.conv.bias[:,None,None]).cpu().data.numpy().reshape(np.prod(self._shape[1:]))
    #         #print(np_y.shape)
    #         np_x,info = sp.sparse.linalg.lgmres(A,np_y,tol=1e-3,maxiter=500,atol=100,M=inv_diag_operator)
    #         assert info==0, f"lgmres failed with info {info}"
    #         x[i] = torch.from_numpy(np_x.astype(np.float32).reshape(y[i].shape)).to(self.conv.weight.device)
    #     return x

    # def log_data(self,logger,step,name=None):
    #     logger.add_scalars('info',{
    #         f'Reverse iters_{name}': self.iters_per_reverse,})
    def np_matvec(self,V):
        self._reverse_iters +=1
        V_pt_img = torch.from_numpy(V.reshape(1,*self._shape[1:]).astype(np.float32)).to(self.conv.weight.device)
        return F.conv2d(V_pt_img,self.conv.weight,padding=1).cpu().data.numpy().reshape(V.shape)
        #return (self(V_pt_img)-self.conv.bias[None,:,None,None]).cpu().data.numpy().reshape(V.shape)
    def logdet(self):
        bs,c,h,w = self._shape
        padded_weight = F.pad(self.conv.weight,(0,h-3,0,w-3))
        w_fft = torch.rfft(padded_weight, 2, onesided=False, normalized=False)
        # pull out real and complex parts
        A = w_fft[...,0]
        B = w_fft[...,1]
        D = torch.cat([ torch.cat([ A, B],dim=1), 
                        torch.cat([-B, A],dim=1)], dim=0).permute(2,3,0,1)
        Dt = D.permute(0, 1, 3, 2) #transpose of D
        lhs = torch.matmul(D, Dt)
        chol_output = torch.cholesky(lhs+3e-5*torch.eye(lhs.size(-1)).to(lhs.device))
        eigs = torch.diagonal(chol_output,dim1=-2,dim2=-1)
        return (eigs.log().sum() / 2.0).expand(bs)


class iElu(nn.ELU):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        self._last_x = x
        return super().forward(x)
    def inverse(self,y):
        # y if y>0 else log(1+y)
        return F.relu(y) - F.relu(-y)*torch.log(1+y)/y
    def logdet(self):
        #logdetJ = \sum_i log J_ii # sum over c,h,w not batch
        return (-F.relu(-self._last_x)).sum(3).sum(2).sum(1) 

def iConvBNelu(ch):
    return iSequential(iConv2d(ch,ch),iBN(ch),iElu())

def passThrough(*layers):
    return iSequential(*[both(layer,I) for layer in layers])



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
            SqueezeLayer(2),
            #Expression(lambda x: torch.cat((x[:,:k],x[:,3*k:]),dim=1)),
            *iConvBNelu(4*k),
            *iConvBNelu(4*k),
            *iConvBNelu(4*k),
            SqueezeLayer(2),
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
            passThrough(SqueezeLayer(2)),
            passThrough(iConv1x1(4*k)),
            keepChannels(2*k),
            
            passThrough(*iConvBNelu(2*k)),
            passThrough(*iConvBNelu(2*k)),
            passThrough(*iConvBNelu(2*k)),
            passThrough(SqueezeLayer(2)),
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
    # def sample(self,bs=1):
    #     z_all = [torch.randn(bs,*shape).to(self.device) for shape in self.z_shapes]
    #     return self.inverse(z_all)

class iEluNetMultiScaleLarger(iEluNetMultiScale):
    def __init__(self, num_classes=10,k=32):
        super().__init__()
        self.num_classes = num_classes
        self.k = k = 2*k
        self.body = iSequential(
            padChannels(k-3),
            addZslot(),
            passThrough(*iConvBNelu(k)),
            passThrough(*iConvBNelu(k)),
            passThrough(*iConvBNelu(k)),
            passThrough(SqueezeLayer(2)),
            passThrough(iConv1x1(4*k)),
            keepChannels(2*k),
            passThrough(*iConvBNelu(2*k)),
            passThrough(*iConvBNelu(2*k)),
            passThrough(*iConvBNelu(2*k)),
            passThrough(SqueezeLayer(2)),
            passThrough(iConv1x1(8*k)),
            keepChannels(2*k),
            passThrough(*iConvBNelu(2*k)),
            passThrough(*iConvBNelu(2*k)),
            passThrough(*iConvBNelu(2*k)),
            passThrough(iConv2d(2*k,2*k)),
            Join(),
        )
        self.head = nn.Sequential(
            nn.BatchNorm2d(2*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(2*k,num_classes)
        )

import unittest



class iConv1x1(nn.Conv2d):
    def __init__(self, channels):
        super().__init__(channels,channels,1)

    def logdet(self):
        bs,c,h,w = self._input_shape
        return (torch.slogdet(self.weight[:,:,0,0])[1]*h*w).expand(bs)
    def inverse(self,y):
        bs,c,h,w = self._input_shape
        inv_weight = torch.inverse(self.weight[:,:,0,0].double()).float().view(c, c, 1, 1)
        debiased_y = y - self.bias[None,:,None,None]
        x = F.conv2d(debiased_y,inv_weight)
        return x

    def forward(self, x):
        self._input_shape = x.shape
        return F.conv2d(x,self.weight,self.bias)


class TestLogDet(unittest.TestCase):
    def test_iconv(self, channels=128, seed=2019,h=8):
        torch.random.manual_seed(seed)

        weight_obj = iConv2d(channels, channels)
        w=h
        input_activation = torch.randn(1,channels,h,w)
        _ = weight_obj(input_activation)
        weight = weight_obj.conv.weight
        weight_numpy = weight.detach().cpu().permute((2,3,0,1)).numpy()

        # compute 2d fft 
       # print(weight_numpy.shape)
        kernel_fft = np.fft.fft2(weight_numpy,[h,w], axes=[0,1], norm=None)
        padded_numpy = np.pad(weight_numpy,((0,h-3),(0,w-3),(0,0),(0,0)),mode='constant')
        kernel_fft2 = np.fft.fft2(padded_numpy, axes=[0,1])
        #print("original",(kernel_fft-kernel_fft2))
        # then take svds
        svds = np.linalg.svd(kernel_fft, compute_uv=False)
        #print(svds)
        #print(svds.shape)
        #print(svds)
        # finally log det is sum(log(singular values))
        true_logdet = np.sum(np.log(svds))
        #print(np.min(svds))
        relative_error = torch.norm(true_logdet - weight_obj.logdet()) / np.linalg.norm(true_logdet)
        print('relative error is: ', relative_error)
        self.assertLess(relative_error, 1e-4)

class TestFFTConv(unittest.TestCase):
    def test_ifftconv(self):
        w=h = 3
        channels = 1

        torch.random.manual_seed(2019)
        input_activation = torch.randn(1,channels,h,w)
        layer = iConv2d(channels,channels)
        weight_numpy = layer.conv.weight.detach().cpu().permute((2,3,0,1)).numpy()
        #print(weight_numpy.shape,w,h)
        padded_numpy = np.pad(weight_numpy,(((w-1)//2,(w-1)//2+(w-1)%2),((w-1)//2,(w-1)//2+(w-1)%2),(0,0),(0,0)),mode='constant')
        kernel_fft = np.conj(np.fft.fft2(padded_numpy, axes=[0,1]))
        #kernel_fft = np.conj(np.fft.rfft2(weight_numpy,[h+2,w+2], axes=[0,1]))
        u,sigma,vh = np.linalg.svd(kernel_fft)
        #print(padded_numpy.shape,kernel_fft.shape,u.shape,vh.shape)
        input_np = input_activation.permute((2,3,1,0)).data.numpy()
        padded_input = np.pad(input_np,((1,1),(1,1),(0,0),(0,0)),mode='constant')
        fft_input = np.fft.fft2(padded_input, axes=[0,1])
        print(kernel_fft.shape,fft_input.shape)
        filtered = (u@((sigma[...,None]*vh)@fft_input))
        output = np.fft.ifft2(filtered,axes=[0,1]).transpose((3,2,0,1))[...,:h,:w]
        #print(np.fft.ifft2(filtered,axes=[0,1]).transpose((3,2,0,1)))
        conv_output = F.conv2d(input_activation,layer.conv.weight,padding=1).data.numpy()
        
        print(output)
        print(conv_output)
        #print(output.shape,conv_output.shape)
        print(np.linalg.norm(output-conv_output)/np.linalg.norm(output))

if __name__ == "__main__":
    unittest.main()