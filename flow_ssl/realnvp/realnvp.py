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
from flow_ssl.invertible.parts import Join
from flow_ssl.invertible.parts import passThrough


class RealNVP(nn.Module):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super(RealNVP, self).__init__()
        
        layers = [addZslot(), passThrough(iLogits())]

        for scale in range(num_scales):
            in_couplings = self._threecouplinglayers(in_channels, mid_channels, num_blocks, MaskCheckerboard)
            layers.append(passThrough(*in_couplings))

            if scale == num_scales - 1:
                layers.append(passThrough(
                    CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True))))
            else:
                layers.append(passThrough(SqueezeLayer(2)))
                out_couplings = self._threecouplinglayers(4 * in_channels, 2 * mid_channels, num_blocks, MaskChannelwise)
                layers.append(passThrough(*out_couplings))
                layers.append(keepChannels(2 * in_channels))
            
            in_channels *= 2
            mid_channels *= 2

        layers.append(Join())
        self.body = iSequential(*layers)
        #print(layers)

    def forward(self,x):
        return self.body(x)

    def logdet(self):
        return self.body.logdet()

    def inverse(self,z):
        return self.body.inverse(z)

    @staticmethod
    def _threecouplinglayers(in_channels, mid_channels, num_blocks, mask_class):
        layers = [
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False)),
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=True)),
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False))
        ]
        return layers
