import torch
import torch.nn.functional as F
import torch.nn as nn

class iSLReLU(nn.Module):
    def __init__(self,slope=.1):
        self.alpha = (1 - slope)/(1+slope)
        super().__init__()
    def forward(self,x):
        self._last_x = x
        y = (x+self.alpha*torch.sqrt(1+x*x))/(1+self.alpha)
        return y
    def inverse(self,y):
        # y if y>0 else log(1+y)
        a = self.alpha
        b = (1+a)*y# + a
        x = (torch.sqrt(a**2 + (a*b)**2-a**4) - b)/(a**2-1)
        #assert not torch.isnan(x).any(), "Nans in iSLReLU"
        return x
    def logdet(self):
        #logdetJ = \sum_i log J_ii # sum over c,h,w not batch
        x = self._last_x
        a = self.alpha
        log_dets = torch.log((1+a*x/(torch.sqrt(1+x*x)))/(1+a))
        if len(x.shape)==2: return log_dets.sum(1)
        else: return log_dets.sum(3).sum(2).sum(1)

class iElu(nn.ELU):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        self._last_x = x
        return super().forward(x)
    def inverse(self,y):
        # y if y>0 else log(1+y)
        x = F.relu(y) - F.relu(-y)*torch.log(1+y)/y
        #assert not torch.isnan(x).any(), "Nans in iElu"
        return x
    def logdet(self):
        #logdetJ = \sum_i log J_ii # sum over c,h,w not batch
        return (-F.relu(-self._last_x)).sum(3).sum(2).sum(1) 