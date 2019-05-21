import numpy as np
from flow_ssl.invertible.coupling_layers import iConv2d,fft_conv3x3,inverse_fft_conv3x3,inverse_fft_conv3x3_pytorch
from flow_ssl.invertible.coupling_layers import phi, phi_inv, phi_vec, phi_inv_vec,Clip_OperatorNorm,Clip_OperatorNorm_PT
import torch
import torch.nn.functional as F
import unittest


class TestLogDet(unittest.TestCase):
    def test_iconv(self, channels=64, seed=2019,h=8):
        torch.random.manual_seed(seed)

        weight_obj = iConv2d(channels, channels)
        w=h
        input_activation = torch.randn(1,channels,h,w)
        _ = weight_obj(input_activation)
        weight = weight_obj.conv.weight
        weight_numpy = weight.detach().cpu().permute((2,3,0,1)).numpy()

        # compute 2d fft 
       # print(weight_numpy.shape)
        kernel_fft = np.fft.fft2(weight_numpy,[h,w], axes=[0,1], norm=None)
        padded_numpy = np.pad(weight_numpy,((0,h-3),(0,w-3),(0,0),(0,0)),mode='constant')
        kernel_fft2 = np.fft.fft2(padded_numpy, axes=[0,1])
        #print("original",(kernel_fft-kernel_fft2))
        # then take svds
        svds = np.linalg.svd(kernel_fft, compute_uv=False)
        # finally log det is sum(log(singular values))
        true_logdet = np.sum(np.log(svds))
        #print(np.min(svds))
        relative_error = torch.norm(true_logdet - weight_obj.logdet()) / np.linalg.norm(true_logdet)
        #print('relative error is: ', relative_error)
        self.assertLess(relative_error, 4e-3)

class TestFFTConv(unittest.TestCase):

    def test_fftconv(self):
        w=h = 3
        channels = 5

        torch.random.manual_seed(2019)
        input_activation = torch.randn(1,channels,h,w)
        layer = iConv2d(channels,channels)
        fft_output = fft_conv3x3(input_activation,layer.conv.weight).data.numpy()
        conv_output = F.conv2d(input_activation,layer.conv.weight,padding=1).data.numpy()
        rel_error = np.linalg.norm(fft_output-conv_output)/np.linalg.norm(fft_output)
        self.assertLess(rel_error, 1e-6)

    def test_ifftconv(self):
        w=h = 8
        channels = 128
        w=h = 3
        channels = 1
        torch.random.manual_seed(2019)
        x = torch.randn(1,channels,h,w)
        layer = iConv2d(channels,channels)
        
        conv_output = layer(x) - layer.conv.bias[None,:,None,None]
        ifft_output = inverse_fft_conv3x3(conv_output,layer.conv.weight)
        #print(ifft_output)
        #print(x)
        rel_error = (ifft_output-x).norm()/x.norm()
        #print(rel_error)
        self.assertLess(rel_error, 1e-4)

    def test_ifftconv_pytorch(self):
        w=h = 8
        channels = 128

        torch.random.manual_seed(2019)
        x = torch.randn(1,channels,h,w)
        layer = iConv2d(channels,channels)
        
        conv_output = layer(x) - layer.conv.bias[None,:,None,None]
        ifft_output = inverse_fft_conv3x3_pytorch(conv_output,layer.conv.weight)
        #print(ifft_output)
        #print(x)
        rel_error = (ifft_output-x).norm()/x.norm()
        #print(rel_error)
        self.assertLess(rel_error, 1e-4)

    def test_clipping_pytorch(self):
        w=h = 16
        channels = 16

        torch.random.manual_seed(2019)
        x = torch.randn(1,channels,h,w).cuda()
        layer = iConv2d(channels,channels).cuda()
        #print(layer.conv.weight.data)
        conv_output = layer(x) - layer.conv.bias[None,:,None,None]
        clipped_np = Clip_OperatorNorm(layer.conv.weight.data,(h,w),(3,10))
        clipped_pt = Clip_OperatorNorm_PT(layer.conv.weight.data,(h,w),(3,10))
        rel_err = (clipped_np-clipped_pt).norm()/clipped_np.norm()
        #layer.conv.weight.data = Clip_OperatorNorm(layer.conv.weight.data,(h,w),(1,None))
        #layer.conv.weight.data = Clip_OperatorNorm(layer.conv.weight.data,(h,w),(1,None))
        #print(ifft_output)
        #print(x)
        # rel_error = (ifft_output-x).norm()/x.norm()
        # #print(rel_error)
        self.assertLess(rel_err, 1e-6)

    def test_phi(self):
        C = torch.randn(5,5,8,8,2)
        Cp = phi_inv(phi(C)) 
        #print("C' shape:",Cp.shape)
        rel_err = (Cp- C).norm()/C.norm()
        self.assertLess(rel_err,1e-8)

    def test_phi_vec(self):
        C = torch.randn(2,5,8,8,2)
        C_p = phi_inv_vec(phi_vec(C))
        #print("C' shape:",C_p.shape)
        rel_err = (C_p - C).norm()/C.norm()
        self.assertLess(rel_err,1e-8)

if __name__ == "__main__":
    unittest.main()