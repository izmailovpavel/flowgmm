import numpy as np
from layers import iConv2d

import torch

import unittest


class TestLogDet(unittest.TestCase):
    def test_iconv(self, size=64, channels=3, seed=2019):
        torch.random.manual_seed(seed)

        weight_obj = iConv2d(64, 64, 3)
        weight = weight_obj.conv.weight
        weight_numpy = weight.detach().cpu().numpy()

        # compute 2d fft 
        kernel_fft = np.fft.fft2(weight_numpy, [weight.size(1), weight.size(0)], axes=[0,1], norm=None)
        # then take svds
        svds = np.linalg.svd(kernel_fft, compute_uv=False)

        # finally log det is sum(log(singular values))
        true_logdet = np.sum(np.log(svds))

        relative_error = torch.norm(true_logdet - weight_obj.logdet()) / np.linalg.norm(true_logdet)
        print('relative error is: ', relative_error)
        self.assertLess(relative_error, 1e-6)

if __name__ == "__main__":
    unittest.main()