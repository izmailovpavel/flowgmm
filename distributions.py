import torch
from torch import distributions
import numpy as np

class SSLGaussMixture(torch.distributions.Distribution):

    def __init__(self,  means, cov_std=1., device=None):
        self.n_components, self.d = means.shape
        cov = cov_std**2 * torch.eye(self.d).to(device)
        means = means.to(device)
        self.gaussians = [distributions.MultivariateNormal(mean, cov)
                          for mean in means]
        
    def sample(self, sample_shape):
        n_samples = sample_shape[0]
        idx = np.random.choice(self.n_components, size=(n_samples, 1))
        all_samples = [g.sample(sample_shape) for g in self.gaussians]
        samples = all_samples[0]
        for i in range(self.n_components):
            mask = np.where(idx == i)
            samples[mask] = all_samples[i][mask]
        return samples
        
    def log_prob(self, x, y=None, label_weight=1.):
        all_probs = [torch.exp(g.log_prob(x)) for g in self.gaussians]
        probs = sum(all_probs) / self.n_components
        log_probs = torch.log(probs)
        for i in range(self.n_components):
            mask = (y == i)
            log_probs[mask] = all_probs[i][mask] * label_weight
        return log_probs

