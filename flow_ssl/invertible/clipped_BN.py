# -*- coding: utf-8 -*-
# File   : batchnorm_reimpl.py
# Author : acgtyrant
# Date   : 11/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import torch
import torch.nn as nn
import torch.nn.init as init


class MeanOnlyBN(nn.BatchNorm2d):
    # See https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        if 'affine' in kwargs:
            assert kwargs['affine'], "only affine supported"

    def forward(self, input_):
        self._check_input_dim(input_)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
        sum_ = input_.sum(1)
        sum_of_square = input_.pow(2).sum(1)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean

        self.running_mean = (
                (1 - exponential_average_factor) * self.running_mean
                + exponential_average_factor * mean.detach()
        )
        unbias_var = sumvar / (numel - 1)
        self.running_var = (
                (1 - exponential_average_factor) * self.running_var
                + exponential_average_factor * unbias_var.detach()
        )
        bias_var = sumvar / numel
        if self.track_running_stats and not self.training:
            mean, bias_var = self.running_mean,self.running_var
        inv_std = 1 / (bias_var + self.eps).pow(0.5).unsqueeze(1)
        mul = torch.min(inv_std * self.weight.unsqueeze(1),torch.ones_like(inv_std))
        output = ((input_ - mean.unsqueeze(1))  + self.bias.unsqueeze(1))

        return output.view(channels, batchsize, height, width).permute(1, 0, 2, 3).contiguous()


class iBN(nn.BatchNorm2d):
    # See https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        if 'affine' in kwargs:
            assert kwargs['affine'], "only affine supported"

    def forward(self, input_):
        self._check_input_dim(input_)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
        sum_ = input_.sum(1)
        sum_of_square = input_.pow(2).sum(1)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean

        self.running_mean = (
                (1 - exponential_average_factor) * self.running_mean
                + exponential_average_factor * mean.detach()
        )
        unbias_var = sumvar / (numel - 1)
        self.running_var = (
                (1 - exponential_average_factor) * self.running_var
                + exponential_average_factor * unbias_var.detach()
        )
        bias_var = sumvar / numel
        if self.track_running_stats and not self.training:
            mean, bias_var = self.running_mean,self.running_var
        inv_std = 1 / (bias_var + self.eps).pow(0.5).unsqueeze(1)
        self.numel,self.height,self.width,self.batchsize = numel,height,width,batchsize
        mul = inv_std * self.weight.unsqueeze(1)
        self.inv_std = inv_std.detach()
        output = ((input_ - mean.unsqueeze(1))*mul  + self.bias.unsqueeze(1))

        return output.view(channels, batchsize, height, width).permute(1, 0, 2, 3).contiguous()

    def logdet(self):
        mul = self.inv_std*self.weight.unsqueeze(1)
        bn_logdet = (torch.log(mul).sum()*self.height*self.width).expand(self.batchsize)
        #print(f"BN logdet: {bn_logdet}")
        return bn_logdet

    def inverse(self,y):
        assert not self.training, "inverse must be computed in eval mode"
        batchsize, channels, height, width = y.size()
        numel = batchsize * height * width
        y_reshaped = y.permute(1,0,2,3).contiguous().view(channels,numel)
        mul = self.weight.unsqueeze(1)/(self.running_var + self.eps).pow(0.5).unsqueeze(1)
        unsquashed_y = (y_reshaped - self.bias.unsqueeze(1))/mul + self.running_mean.unsqueeze(1)
        x = unsquashed_y.view(channels,batchsize,height,width).permute(1,0,2,3).contiguous()
        #assert not torch.isnan(x).any(), "Nans in iBN"
        return x