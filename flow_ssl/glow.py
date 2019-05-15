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
from flow_ssl.invertible.coupling_layers import iConv1x1


class Glow(nn.Module):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super(RealNVP, self).__init__()
        
        layers = [addZslot(), passThrough(iLogits())]

        for scale in range(num_scales):
                
            num_in = 4 if scale == num_scales-1 else 3
            layers += [passThrough(*self._glow_step(in_channels, mid_channels, num_blocks) 
                       for _ in range(3)]
            layers.append(passThrough(SqueezeLayer(2)))
            layers += [passThrough(*self._glow_step(4 * in_channels, 2 * mid_channels, num_blocks) 
                       for _ in range(3)]
            layers.append(keepChannels(2 * in_channels))
            
            in_channels *= 2
            mid_channels *= 2

        layers.append(FlatJoin())
        self.body = iSequential(*layers)
        #print(layers)

    def forward(self,x):
        return self.body(x)

    def logdet(self):
        return self.body.logdet()

    def inverse(self,z):
        return self.body.inverse(z)

    @staticmethod
    def _glowstep(in_channels, mid_channels, num_blocks):
        layers = [
                #TODO: actnorm
                iConv1x1(in_channels),
                CouplingLayer(in_channels, mid_channels, num_blocks, MaskChannelwise(reverse_mask=False)),
        ]
        return layers
