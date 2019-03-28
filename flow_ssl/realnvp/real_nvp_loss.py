import numpy as np
import torch.nn as nn

class RealNVPLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, prior, k=256):
        super(RealNVPLoss, self).__init__()
        self.k = k
        self.prior = prior

    def forward(self, z, sldj, y=None):
        #PAVEL: need to figure out the shapes
        print("real_nvp_loss.py, RealNVPLoss, forward, z.shape",
              z.shape)
        if y is not None:
            prior_ll = self.prior.log_prob(z, y)
        else:
            prior_ll = self.prior.log_prob(z)
        print("real_nvp_loss.py, RealNVPLoss, forward, prior_ll.shape, z.shape",
              prior_ll.shape, z.shape)
        prior_ll -= np.log(self.k) * np.prod(z.size()[1:]) 
        #PAVEL: this doesn't even affect the gradient, right?
        #PAVEL: I guess this is supposed to compute bits per dimension. Check

        ll = prior_ll + sldj
        nll = -ll.mean()

        return nll
