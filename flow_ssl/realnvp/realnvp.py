import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_ssl.realnvp.coupling_layer import CouplingLayer
from flow_ssl.realnvp.coupling_layer import CouplingLayerTabular
from flow_ssl.realnvp.coupling_layer import MaskCheckerboard
from flow_ssl.realnvp.coupling_layer import MaskChannelwise
from flow_ssl.realnvp.coupling_layer import MaskTabular

from flow_ssl.invertible import iSequential
from flow_ssl.invertible.downsample import iLogits
from flow_ssl.invertible.downsample import keepChannels
from flow_ssl.invertible.downsample import SqueezeLayer
from flow_ssl.invertible.parts import addZslot
from flow_ssl.invertible.parts import FlatJoin
from flow_ssl.invertible.parts import passThrough
from flow_ssl.invertible.coupling_layers import iConv1x1
from flow_ssl.invertible import Swish, ActNorm1d, ActNorm2d

class RealNVPBase(nn.Module):

    def forward(self,x):
        return self.body(x)

    def logdet(self):
        return self.body.logdet()

    def inverse(self,z):
        return self.body.inverse(z)

#TODO: batchnorm?
class RealNVP(RealNVPBase):

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

        layers.append(FlatJoin())
        self.body = iSequential(*layers)
        #print(layers)

    @staticmethod
    def _threecouplinglayers(in_channels, mid_channels, num_blocks, mask_class):
        layers = [
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False)),
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=True)),
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False))
        ]
        return layers


class RealNVPw1x1(RealNVP):
    @staticmethod
    def _threecouplinglayers(in_channels, mid_channels, num_blocks, mask_class):
        layers = [
                iConv1x1(in_channels),
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False)),
                iConv1x1(in_channels),
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=True)),
                iConv1x1(in_channels),
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False))
        ]
        return layers



class RealNVPw1x1ActNorm(RealNVP):
    @staticmethod
    def _threecouplinglayers(in_channels, mid_channels, num_blocks, mask_class):
        layers = [
                ActNorm2d(in_channels),
                iConv1x1(in_channels),
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False)),
                ActNorm2d(in_channels),
                iConv1x1(in_channels),
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=True)),
                ActNorm2d(in_channels),
                iConv1x1(in_channels),
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False))
        ]
        return layers


class RealNVPMNIST(RealNVPBase):
    def __init__(self, in_channels=1, mid_channels=64, num_blocks=4):
        super(RealNVPMNIST, self).__init__()
        
        self.body = iSequential(
                addZslot(), 
                passThrough(iLogits()),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False))),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True))),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False))),
                passThrough(SqueezeLayer(2)),
                passThrough(CouplingLayer(4*in_channels, mid_channels, num_blocks, MaskChannelwise(reverse_mask=False))),
                passThrough(CouplingLayer(4*in_channels, mid_channels, num_blocks, MaskChannelwise(reverse_mask=True))),
                passThrough(CouplingLayer(4*in_channels, mid_channels, num_blocks, MaskChannelwise(reverse_mask=False))),
                keepChannels(2*in_channels),                                                      
                passThrough(CouplingLayer(2*in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False))),
                passThrough(CouplingLayer(2*in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True))),
                passThrough(CouplingLayer(2*in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False))),
                passThrough(SqueezeLayer(2)),
                passThrough(CouplingLayer(8*in_channels, mid_channels, num_blocks, MaskChannelwise(reverse_mask=False))),
                passThrough(CouplingLayer(8*in_channels, mid_channels, num_blocks, MaskChannelwise(reverse_mask=True))),
                passThrough(CouplingLayer(8*in_channels, mid_channels, num_blocks, MaskChannelwise(reverse_mask=False))),
                keepChannels(4*in_channels),
                passThrough(CouplingLayer(4*in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False))),
                passThrough(CouplingLayer(4*in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True))),
                passThrough(CouplingLayer(4*in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False))),
                passThrough(CouplingLayer(4*in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True))),
                FlatJoin()
            )


class RealNVPTabular(RealNVPBase):

    def __init__(self, in_dim=2, num_coupling_layers=6, hidden_dim=256, 
                 num_layers=2):

        super(RealNVPTabular, self).__init__()
        
        self.body = iSequential(*[
                        CouplingLayerTabular(in_dim, hidden_dim, num_layers, MaskTabular(reverse_mask=bool(i%2)))
                        for i in range(num_coupling_layers)
                    ])
