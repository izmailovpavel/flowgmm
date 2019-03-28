import torch
from torch import distributions, nn
import torch.nn.functional as F
import numpy as np

class SSLGaussMixture(torch.distributions.Distribution):

    def __init__(self, means, cov_stds=None, device=None):
        self.n_components, self.d = means.shape
        self.means = means.to(device)
        self.cov_stds = cov_stds
        if self.cov_stds is None:
            self.cov_stds = torch.ones((len(means)))
        self.device = device

    @property
    def gaussians(self):
        eye = torch.eye(self.d).to(self.device)
        self.gaussians = [distributions.MultivariateNormal(mean, cov_std * eye)
                          for mean, cov_std in zip(self.means, self.cov_stds)]

    def parameters(self):
       return [self.means, self.cov_std]
        
    def sample(self, sample_shape, gaussian_id=None):
        if gaussian_id is not None:
            g = self.gaussians[gaussian_id]
            samples = g.sample(sample_shape)
        else:
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
        if y is not None:
            for i in range(self.n_components):
                mask = (y == i)
                log_probs[mask] = torch.log(all_probs[i][mask]) * label_weight
        return log_probs


#PAVEL: remove later
class SSLGaussMixtureClassifier(SSLGaussMixture):
    
    def __init__(self, means, cov_std=1., device=None):
        super().__init__(means, cov_std, device)
        self.classifier = nn.Sequential(nn.Linear(self.d, self.n_components))

    def parameters(self):
       return self.classifier.parameters() 

    def forward(self, x):
        return self.classifier.forward(x)

    def log_prob(self, x, y, label_weight=1.):
        all_probs = [torch.exp(g.log_prob(x)) for g in self.gaussians]
        probs = sum(all_probs) / self.n_components
        x_logprobs = torch.log(probs)

        mask = (y != -1)
        labeled_x, labeled_y = x[mask], y[mask].long()
        preds = self.forward(labeled_x)
        y_logprobs = F.cross_entropy(preds, labeled_y)

        return x_logprobs - y_logprobs
