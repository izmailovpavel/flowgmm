"""
Code adapted from https://github.com/chrischute/real-nvp, which is in turn
adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import os
import sys
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import utils
import numpy as np
from scipy.spatial.distance import cdist
from torch.nn import functional as F

from flow_ssl.realnvp import RealNVP, RealNVPLoss
from tqdm import tqdm
from flow_ssl.distributions import SSLGaussMixture


def get_class_means(net, trainloader):
    with torch.no_grad():
        means = torch.zeros((10, 3, 32, 32))
        n_batches = 0
        with tqdm(total=len(trainloader.dataset)) as progress_bar:
            n_batches = len(trainloader)
            for x, y in trainloader:
                z, _ = net(x, reverse=False)
                for i in range(10):
                    means[i] += z[y == i].sum(dim=0).cpu() 
                    #PAVEL: not the right way of computing means
                progress_bar.set_postfix(max_mean=torch.max(means), 
                                         min_mean=torch.min(means))
                progress_bar.update(x.size(0))
        means /= 5000
        return means

def sample(net, prior, batch_size, cls, device):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    #z = torch.randn((batch_size, 3, 32, 32), dtype=torch.float32, device=device)
    with torch.no_grad():
        z = prior.sample((batch_size,), gaussian_id=cls)
        z = z.reshape((batch_size, 3, 32, 32))
        x, _ = net(z, reverse=True)
        x = torch.sigmoid(x)

        return x


def test(net, testloader, device, loss_fn, num_samples):
    predictions = list()
    targets = list()
    net.eval()

    nll_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()

    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, y in testloader:

                x = x.to(device)
                y = y.to(device)
                z, sldj = net(x, reverse=False)

                logits = loss_fn.prior.class_logits(z.reshape((len(z), -1)))
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                nll = F.cross_entropy(logits, y)
                acc = (preds == y).float().mean().item()

                acc_meter.update(acc, x.size(0))
                nll_meter.update(nll, x.size(0))

                predictions.append(probs.detach().cpu().numpy())
                targets.append(y.detach().cpu().numpy())

                progress_bar.update(x.size(0))
                progress_bar.set_postfix(nll=nll_meter.avg,
                                     acc=acc_meter.avg)

    return np.vstack(predictions), np.concatenate(targets)


parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                help='path to datasets location (default: None)')
parser.add_argument('--ckpt', type=str, default=None, required=True, metavar='PATH',
                help='path to ckpt (default: None)')
parser.add_argument('--savedir', type=str, default=None, required=True, metavar='PATH',
                help='path to save directory (default: None)')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
parser.add_argument('--num_samples', default=50, type=int, help='Number of samples at test time')
parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')

parser.add_argument('--means', 
                    choices=['from_data', 'pixel_const', 'split_dims', 'split_dims_v2', 'random'], 
                    default='split_dims')
parser.add_argument('--means_r', default=1., type=float,
                    help='r constant used when defyning means')
parser.add_argument('--cov_std', default=1., type=float,
                    help='covariance std for the latent distribution')
parser.add_argument('--save_freq', default=25, type=int, 
                    help='frequency of saving ckpts')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'

# Note: No normalization applied, since RealNVP expects inputs in (0, 1).
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Model
print('Building model...')
net = RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net, args.gpu_ids)
    cudnn.benchmark = True

# Load checkpoint.
print('Resuming from checkpoint at', args.ckpt)
checkpoint = torch.load(args.ckpt)
net.load_state_dict(checkpoint['net'])

#PAVEL: we need to find a good way of placing the means
D = (32 * 32 * 3)
r = args.means_r 
means = torch.zeros((10, D)).to(device)
cov_std = torch.ones((10)) * args.cov_std
cov_std = cov_std.to(device)

if args.means == "from_data":
    print("Computing the means")
    means = get_class_means(net, trainloader)
    means = means.reshape((10, -1)).to(device)

elif args.means == "pixel_const":
    for i in range(10):
        means[i, :] = r * (i-4)

elif args.means == "split_dims":
    mean_portion = D // 10
    for i in range(10):
        means[i, i*mean_portion:(i+1)*mean_portion] = r
elif args.means == "split_dims_v2":
    means = means.reshape((10, 3, 32, 32))
    for c in range(3):
        if c == 2:
            per_channel = 4
        else:
            per_channel = 3
        mean_portion = 32 // per_channel
        for i in range(per_channel):
            means[c * 3 + i, c, i*mean_portion:(i+1)*mean_portion, :] = r
    means = means.reshape((10, -1))
elif args.means == "random":
    for i in range(10):
        means[i] = r * torch.randn(D)
else:
    raise NotImplementedError(args.means)


print("Means:", means)
print("Cov std:", cov_std)
means_np = means.cpu().numpy()
print("Pairwise dists:", cdist(means_np, means_np))

prior = SSLGaussMixture(means, device=device)
loss_fn = RealNVPLoss(prior)

preds, targets = test(net, testloader, device, loss_fn, args.num_samples)

os.makedirs(os.path.join(args.savedir, 'samples'), exist_ok=True)

np.savez(os.path.join(args.savedir, "preds"),
        probs=preds, targets=targets)

# Save samples and data
images = []
for i in range(10):
    images_cls = sample(net, loss_fn.prior, args.num_samples // 10, cls=i, device=device)
    images.append(images_cls)
images = torch.cat(images)
images_concat = torchvision.utils.make_grid(images, nrow=args.num_samples //  10 , padding=2, pad_value=255)
torchvision.utils.save_image(images_concat, 
                            os.path.join(args.savedir, 'samples/eval.png'))
