# ResNet generator and discriminator
from torch import nn
import torch
import torch.nn.functional as F
import torchcontrib
import torchcontrib.nn.functional as contrib
import numpy as np
#from .spectral_normalization import SpectralNorm
#from torch.nn.utils import spectral_norm
from oil.utils.utils import Expression
from oil.architecture.img_gen.ganBase import GanBase, add_spectral_norm, xavier_uniform_init
from oil.architecture.parts import ResBlock, conv2d


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise

class ConstantInput(nn.Module):
    def __init__(self, channels, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channels, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out



class CategoricalFiLM(nn.Module):
    def __init__(self,num_classes,channels):
        super().__init__()
        self.gammas = nn.Embedding(num_classes,channels)
        self.betas = nn.Embedding(num_classes,channels)
    def forward(self,x,y):
        return contrib.film(x,self.gammas(y),self.betas(y))

class Generator(GanBase):
    def __init__(self, num_classes,z_dim=128,img_channels=3,k=128):
        super().__init__(z_dim,img_channels)
        self.num_classes = num_classes
        self.k = k
        self.linear1 = nn.Linear(z_dim, 4 * 4 * k)
        self.res1 = cResBlockGenerator(k,k,num_classes,stride=2)
        self.res2 = cResBlockGenerator(k,k,num_classes,stride=2)
        self.res3 = cResBlockGenerator(k,k,num_classes,stride=2)
        self.final = nn.Sequential(
            nn.BatchNorm2d(k),
            nn.ReLU(),
            nn.Conv2d(k, img_channels, 3, stride=1, padding=1),
            nn.Tanh())
        self.apply(xavier_uniform_init)

    def forward(self,y,z=None):
        if z is None: z = self.sample_z(y.shape[0])
        z = self.linear1(z).view(-1,self.k,4,4)
        z = self.res1(z,y)
        z = self.res2(z,y)
        z = self.res3(z,y)
        return self.final(z)

    def sample_y(self,n=1):
        return (torch.LongTensor(n).random_()%self.num_classes).to(self.device)
        
    def sample(self, n=1):
        return self(self.sample_y(n),self.sample_z(n))


class Discriminator(nn.Module):
    def __init__(self,num_classes,img_channels=3,k=128,out_size=1):
        super().__init__()
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.k = k
        self.phi = nn.Sequential(
                conv2d(img_channels,k),
                ResBlockDiscriminator(k, k, stride=2),
                ResBlockDiscriminator(k, k, stride=2),
                ResBlockDiscriminator(k, k),
                ResBlockDiscriminator(k, k),
                nn.ReLU(),
                Expression(lambda u: u.mean(-1).mean(-1)),
            )
        self.psi = nn.Linear(k, out_size)
        self.apply(add_spectral_norm)
        self.label_embedding = nn.Embedding(num_classes,k)
        self.apply(xavier_uniform_init)
        
    def forward(self, x, y):
        embedded_labels = self.label_embedding(y)
        phi = self.phi(x)
        return self.psi(phi) + (embedded_labels*phi).sum(-1)

class cResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes, stride=1):
        super().__init__()
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.film1 = CategoricalFiLM(num_classes,in_channels) # should it be shared?
        self.relu1 = nn.ReLU()
        self.conv1 = conv2d(in_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.film2 = CategoricalFiLM(num_classes,out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = conv2d(out_channels, out_channels)
        if stride!=1 or in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=stride),
                conv2d(in_channels,out_channels))
        else: 
            self.shortcut = nn.Sequential()

    def forward(self, x,y):
        z = x
        z = self.relu1(self.film1(self.bn1(z),y))
        z = self.conv1(F.interpolate(z,scale_factor=self.stride))
        z = self.conv2(self.relu2(self.film2(self.bn2(z),y)))
        return z + self.shortcut(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()
        modules =  [nn.ReLU(),
                    conv2d(in_channels, out_channels),
                    nn.ReLU(),
                    conv2d(out_channels, out_channels)]
        bypass =   []
        if stride!=1:
            modules += [nn.AvgPool2d(2, stride=stride, padding=0)]
            bypass  += [conv2d(in_channels, out_channels,1),
                        nn.AvgPool2d(2, stride=stride, padding=0)]
        self.model = nn.Sequential(*modules)
        self.bypass = nn.Sequential(*bypass)
    def forward(self, x):
        return self.model(x) + self.bypass(x)