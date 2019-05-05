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
from tqdm import tqdm
from tensorboardX import SummaryWriter

from flow_ssl.realnvp import RealNVP 
from flow_ssl import FlowLoss
from flow_ssl.distributions import SSLGaussMixture
from flow_ssl.data import make_sup_data_loaders
from flow_ssl.data import NO_LABEL
from flow_ssl.data import TransformTwice


def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.sum(np.log(ps))
    return nll


def test(net, testloader, device, loss_fn):
    net.eval()
    ys_lst = []
    preds_lst = []
    probs_lst = []
    probs_x_lst = []
    total_loss = 0.
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, y in testloader:
                x = x.to(device)
                y = y.to(device).cpu().numpy().reshape((-1, 1))
                z, sldj = net(x, reverse=False)

                loss = loss_fn(z, sldj=sldj)
                total_loss += loss * x.size(0)
                z = z.reshape((len(z), -1))

                probs_x = loss_fn.prior.log_prob(z) + sldj
                probs_x = probs_x.cpu().numpy()
                probs_x = probs_x.reshape((-1, 1))

                probs = loss_fn.prior.class_probs(z).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                preds = preds.reshape((-1, 1))

                ys_lst.append(y)
                preds_lst.append(preds)
                probs_lst.append(probs)
                probs_x_lst.append(probs_x)

                progress_bar.update(x.size(0))
                
    ys = np.vstack(ys_lst)
    probs = np.vstack(probs_lst)
    probs_x = np.vstack(probs_x_lst)
    preds = np.vstack(preds_lst)
    loss = total_loss / len(ys)
    acc = (ys == preds).mean()
    bpd = utils.bits_per_dim(x, loss)

    return acc, bpd, loss, ys, probs, probs_x, preds
    

parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

parser.add_argument('--dataset', type=str, default="CIFAR10", required=True, metavar='DATA',
                help='Dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                help='path to datasets location (default: None)')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
parser.add_argument('--num_samples', default=50, type=int, help='Number of samples at test time')
parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
parser.add_argument('--ckpt', type=str, default=None, required=True, metavar='PATH',
                help='path to checkpoint to resume from (default: None)')
parser.add_argument('--filename', type=str, default="preds.npz", required=False, metavar='PATH',
                help='filename to save preds')

# PAVEL
parser.add_argument('--cov_std', default=1., type=float,
                    help='covariance std for the latent distribution')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
start_epoch = 0

# Note: No normalization applied, since RealNVP expects inputs in (0, 1).

transform_test = []
if args.dataset.lower() == "mnist":
    img_shape = (1, 28, 28)
if args.dataset.lower() == "notmnist":
    img_shape = (1, 28, 28)
    transform_test = [transforms.Grayscale()]
elif args.dataset.lower() == "cifar10":
    img_shape = (3, 32, 32)
else:
    raise ValueError("Unknown dataset")

transform_test = transforms.Compose([
    *transform_test,
    transforms.ToTensor()
])

_, testloader, _ = make_sup_data_loaders(
        args.data_path, 
        args.batch_size, 
        args.num_workers, 
        transform_test, 
        transform_test, 
        use_validation=False, 
        shuffle_train=True,
        dataset=args.dataset.lower())

# Model
print('Building model...')
net = RealNVP(num_scales=2, in_channels=img_shape[0], mid_channels=64, num_blocks=8)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net, args.gpu_ids)
    cudnn.benchmark = True #args.benchmark

print('Resuming from checkpoint at', args.ckpt)
checkpoint = torch.load(args.ckpt)
net.load_state_dict(checkpoint['net'])
pred_path = "/".join(args.ckpt.split("/")[:-1])
pred_path = os.path.join(pred_path, args.filename)
print(pred_path)

#PAVEL: we need to find a good way of placing the means
cov_std = torch.ones((10)) * args.cov_std
cov_std = cov_std.to(device)
means = checkpoint['means']


print("Means:", means)
print("Cov std:", cov_std)
means_np = means.cpu().numpy()
print("Pairwise dists:", cdist(means_np, means_np))

prior = SSLGaussMixture(means, device=device)
loss_fn = FlowLoss(prior)

acc, bpd, loss, ys, probs, probs_x, preds = test(net, testloader, "cuda", loss_fn)
print("Accuracy:", acc)
print("BPD:", bpd)
print("Loss:", loss)
print("NLL:", nll(probs, ys))

np.savez(pred_path, ys=ys, probs=probs, probs_x=probs_x)
