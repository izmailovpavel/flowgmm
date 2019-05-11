import torch
import numpy as np
from gyptorch.lazy import LazyTensor
from torch.autograd import Function

class AutogradGpyLogdet(Function):
    @staticmethod
    def forward(ctx,x,y):
        
    def backward(ctx,grad_output):
        pass

def autograd_gpytorch_logdet(x,y):
    J = JacobianLazyTensor(x,y)
    JtJ = J.t()@J
    logdet = 

class JacobianLazyTensor(LazyTensor):
    def __init__(self,inp,output):
        """ Computes the jacobian vector products of the module, with whatever
            has been passed through it"""
        super().__init__()
        self.inp = inp
        self.output = output

    def _matmul(self,rhs):
        with torch.autograd.enable_grad():
            v = rhs # proper reshaping?
            Jv = torch.autograd.grad(self.output,[self.inp],[v],create_graph=True)
        return Jv
    def _t_matmul(self,rhs):
        with torch.autograd.enable_grad():
            u = rhs
            v = torch.ones_like(self.output)
            Jv = torch.autograd.grad(self.output,[self.inp],[v],create_graph=True)
            uJ = torch.autograd.grad(Jv,[v],[u],create_graph=True)
        return uJ
    def _size(self):
        bs = self.output.shape[0]
        d_in = np.product(self.inp.shape[1:])
        d_out = np.product(self.output.shape[1:])
        return torch.Size((bs,d_in,d_out))
    def _transpose_nonbatch(self):
        pass
    def _quad_form_derivative(self):
        pass
    