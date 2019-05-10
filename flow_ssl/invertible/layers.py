
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


class iConv2d(nn.Module):
    """ wraps conv2d in a module with an inverse function """
    def __init__(self,*args,inverse_tol=1e-7,circ=True,**kwargs):
        super().__init__()
        self.conv = conv2d(*args,**kwargs)
        self.inverse_tol = inverse_tol
        self._reverse_iters = 0
        self._inverses_evaluated = 0
        self._circ= circ
    @property
    def iters_per_reverse(self):
        return self._reverse_iters/self._inverses_evaluated

    def forward(self,x):
        self._shape = x.shape
        if self._circ:
            padded_x = pad_circular_nd(x,1,dim=[2,3])
            return F.conv2d(padded_x,self.conv.weight,self.conv.bias)
        else:
            return self.conv(x)
    # FFT inverse method

    def inverse(self,y):
        x = inverse_fft_conv3x3(y-self.conv.bias[None,:,None,None],self.conv.weight)
        return x

    def np_matvec(self,V):
        self._reverse_iters +=1
        V_pt_img = torch.from_numpy(V.reshape(1,*self._shape[1:]).astype(np.float32)).to(self.conv.weight.device)
        return F.conv2d(V_pt_img,self.conv.weight,padding=1).cpu().data.numpy().reshape(V.shape)

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
        x = F.relu(y) - F.relu(-y)*torch.log(1+y)/y
        #assert not torch.isnan(x).any(), "Nans in iElu"
        return x
    def logdet(self):
        #logdetJ = \sum_i log J_ii # sum over c,h,w not batch
        return (-F.relu(-self._last_x)).sum(3).sum(2).sum(1) 


class iSLReLU(nn.Module):
    def __init__(self,slope=.1):
        self.alpha = (1 - slope)/(1+slope)
        super().__init__()
    def forward(self,x):
        self._last_x = x
        y = (x+self.alpha*(torch.sqrt(1+x*x)-1))/(1+self.alpha)
        return y
    def inverse(self,y):
        # y if y>0 else log(1+y)
        a = self.alpha
        b = (1+a)*y + a
        x = (torch.sqrt(a**2 + (a*b)**2-a**4) - b)/(a**2-1)
        #assert not torch.isnan(x).any(), "Nans in iSLReLU"
        return x
    def logdet(self):
        #logdetJ = \sum_i log J_ii # sum over c,h,w not batch
        x = self._last_x
        a = self.alpha
        return (1+a*x/(torch.sqrt(1+x*x))).sum(3).sum(2).sum(1)/(1+a)


def iConvBNelu(ch):
    return iSequential(iConv2d(ch,ch),iBN(ch),iSLReLU())


def passThrough(*layers):
    return iSequential(*[both(layer,I) for layer in layers])


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
