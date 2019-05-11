import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_ssl.realnvp.coupling_layer import CouplingLayer
from flow_ssl.realnvp.coupling_layer import MaskCheckerboard
from flow_ssl.realnvp.coupling_layer import MaskChannelwise

from flow_ssl.invertible.auto_inverse import iSequential
from flow_ssl.invertible.layers import keepChannels
from flow_ssl.invertible.layers import addZslot
from flow_ssl.invertible.layers import passThrough


for coupling in self.in_couplings:
    x, sldj = coupling(x, sldj, reverse)

if not self.is_last_block:
    # Squeeze -> 3x coupling (channel-wise)
    x = squeeze_2x2(x, reverse=False)
    for coupling in self.out_couplings:
        x, sldj = coupling(x, sldj, reverse)
    x = squeeze_2x2(x, reverse=True)

    # Re-squeeze -> split -> next block
    x = squeeze_2x2(x, reverse=False, alt_order=True)
    x, x_split = x.chunk(2, dim=1)
    x, sldj = self.next_block(x, sldj, reverse)
    x = torch.cat((x, x_split), dim=1)
    x = squeeze_2x2(x, reverse=True, alt_order=True)


class RealNVPBase(nn.Module):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super(RealNVP, self).__init__()
        # Register data_constraint to pre-process images, not learnable
        self.register_buffer('data_constraint', torch.tensor([0.9], dtype=torch.float32)) 
        
        layers = [addZslot(), passThrough(Dequantization(self.data_constraint))]

        for scale in range(num_scales):
            in_couplings = *_threecouplinglayers(in_channels, mid_channels, num_blocks, MaskCheckerboard)
            layers.append(passThrough(*in_couplings))

            if scale == num_scales - 1:
                layers.append(passThrough(
                    CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True))))
            else:
                layers.append(passThrough(SqueezeLayer(2)))
                out_couplings = _threecouplinglayers(4 * in_channels, 2 * mid_channels, num_blocks, MaskChannelWise)
                layers.append(passThrough(*out_couplings))
                layers.append(keepChannels(2 * in_channels))

            
            in_channels *= 2
            mid_channels *= 2
        self.body = iSequential(*layers)
        print(layers)

    def forward(self,x):
        z = self.body(x)
        return self.head(z)

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

    ##PAVEL: turn this into a layer
    #def _pre_process(self, x):
    #    """Dequantize the input image `x` and convert to logits.

    #    Args:
    #        x (torch.Tensor): Input image.

    #    Returns:
    #        y (torch.Tensor): Dequantized logits of `x`.

    #    See Also:
    #        - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
    #        - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
    #    """
    #    y = (x * 255. + torch.rand_like(x)) / 256.
    #    y = (2 * y - 1) * self.data_constraint
    #    y = (y + 1) / 2
    #    y = y.log() - (1. - y).log()

    #    # Save log-determinant of Jacobian of initial transform
    #    ldj = F.softplus(y) + F.softplus(-y) \
    #        - F.softplus((1. - self.data_constraint).log() - self.data_constraint.log())
    #    sldj = ldj.view(ldj.size(0), -1).sum(-1)

    #    return y, sldj


class _RealNVP(nn.Module):
    """Recursive builder for a `RealNVP` model.

    Each `_RealNVPBuilder` corresponds to a single scale in `RealNVP`,
    and the constructor is recursively called to build a full `RealNVP` model.

    Args:
        scale_idx (int): Index of current scale.
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
    """
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super(_RealNVP, self).__init__()

        self.is_last_block = (scale_idx == num_scales - 1)

        # 3 coupling layers with alternating checkerboard masks
        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False)
        ])

        if self.is_last_block:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True))
        else:
            # 3 coupling layers with alternating channel-wise mask
            self.out_couplings = nn.ModuleList([
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=True),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False)
            ])
            # PAVEL: why is this code recursive?
            self.next_block = _RealNVP(scale_idx + 1, num_scales, 2 * in_channels, 2 * mid_channels, num_blocks)

    def forward(self, x, sldj, reverse=False):
        if reverse:
            if not self.is_last_block:
                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)

                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2(x, reverse=False)
                for coupling in reversed(self.out_couplings):
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)

            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, reverse)
        else:
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj, reverse)

            if not self.is_last_block:
                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2(x, reverse=False)
                for coupling in self.out_couplings:
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)

                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)

        return x, sldj
