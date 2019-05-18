import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_ssl.realnvp.coupling_layer import CouplingLayer
from flow_ssl.realnvp.coupling_layer import MaskCheckerboard
from flow_ssl.realnvp.coupling_layer import MaskChannelwise

from flow_ssl.invertible import iSequential
from flow_ssl.invertible.downsample import iLogits
from flow_ssl.invertible.downsample import keepChannels
from flow_ssl.invertible.downsample import SqueezeLayer
from flow_ssl.invertible.parts import addZslot
from flow_ssl.invertible.parts import FlatJoin
from flow_ssl.invertible.parts import passThrough
from flow_ssl.invertible.parts import ActNorm
from flow_ssl.invertible.coupling_layers import iConv1x1


class GlowBase(nn.Module):
    
    def __init__(self):
        super(GlowBase, self).__init__()

    def forward(self,x):
        return self.body(x)

    def logdet(self):
        return self.body.logdet()

    def inverse(self,z):
        return self.body.inverse(z)

    @staticmethod
    def _glow_step(in_channels, mid_channels, num_blocks):
        layers = [
                ActNorm(in_channels),
                iConv1x1(in_channels),
                CouplingLayer(in_channels, mid_channels, num_blocks, MaskChannelwise(reverse_mask=False)),
        ]
        return layers


class Glow(GlowBase):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super(Glow, self).__init__()
        
        layers = [addZslot(), passThrough(iLogits())]

        for scale in range(num_scales):
            num_in = 4 if scale == num_scales-1 else 3
            for _ in range(num_in):
                layers.append(passThrough(*self._glow_step(in_channels, mid_channels, num_blocks)))
            layers.append(passThrough(SqueezeLayer(2)))
            num_out = 0 if scale == num_scales-1 else 3
            for _ in range(num_in):
                layers.append(passThrough(*self._glow_step(4*in_channels, 2*mid_channels, num_blocks)))
            layers.append(keepChannels(2 * in_channels))
            
            in_channels *= 2
            mid_channels *= 2

        layers.append(FlatJoin())
        self.body = iSequential(*layers)


class GlowMNIST(GlowBase):
    def __init__(self, in_channels=1, mid_channels=64, num_blocks=4):
        super(GlowMNIST, self).__init__()
        
        self.body = iSequential(
                addZslot(), 
                passThrough(iLogits()),
                passThrough(*self._glow_step(in_channels, mid_channels, num_blocks, MaskCheckerboard)),
                passThrough(*self._glow_step(in_channels, mid_channels, num_blocks, MaskCheckerboard)),
                passThrough(*self._glow_step(in_channels, mid_channels, num_blocks, MaskCheckerboard)),
                passThrough(SqueezeLayer(2)),
                passThrough(*self._glow_step(4*in_channels, mid_channels, num_blocks, MaskChannelwise)),
                passThrough(*self._glow_step(4*in_channels, mid_channels, num_blocks, MaskChannelwise)),
                keepChannels(2*in_channels),
                passThrough(*self._glow_step(2*in_channels, mid_channels, num_blocks, MaskCheckerboard)),
                passThrough(*self._glow_step(2*in_channels, mid_channels, num_blocks, MaskCheckerboard)),
                passThrough(*self._glow_step(2*in_channels, mid_channels, num_blocks, MaskCheckerboard)),
                passThrough(SqueezeLayer(2)),
                passThrough(*self._glow_step(8*in_channels, mid_channels, num_blocks, MaskChannelwise)),
                passThrough(*self._glow_step(8*in_channels, mid_channels, num_blocks, MaskChannelwise)),
                keepChannels(4*in_channels),
                passThrough(*self._glow_step(4*in_channels, mid_channels, num_blocks, MaskCheckerboard)),
                passThrough(*self._glow_step(4*in_channels, mid_channels, num_blocks, MaskCheckerboard)),
                passThrough(*self._glow_step(4*in_channels, mid_channels, num_blocks, MaskCheckerboard)),
                passThrough(*self._glow_step(4*in_channels, mid_channels, num_blocks, MaskChannelwise)),
                FlatJoin()
            )

    @staticmethod
    def _glow_step(in_channels, mid_channels, num_blocks, mask_class):
        layers = [
                ActNorm(in_channels),
                iConv1x1(in_channels),
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False)),
        ]
        return layers
