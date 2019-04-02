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
from flow_ssl.realnvp import FlowLoss
from flow_ssl.distributions import SSLGaussMixture
from flow_ssl.data import make_sup_data_loaders


def schedule(epoch):
    lr_ratio = 0.01
    t = epoch / args.num_epochs
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr * factor


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


def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm, writer):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = utils.AverageMeter()
    loss_unsup_meter = utils.AverageMeter()
    loss_nll_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            z, sldj = net(x, reverse=False)

            logits = loss_fn.prior.class_logits(z.reshape((len(z), -1)))
            loss_nll = F.cross_entropy(logits, y)

            loss_unsup = loss_fn(z, sldj=sldj)
            loss = loss_nll + loss_unsup

            loss.backward()
            utils.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean().item()

            acc_meter.update(acc, x.size(0))
            loss_meter.update(loss.item(), x.size(0))
            loss_unsup_meter.update(loss_unsup.item(), x.size(0))
            loss_nll_meter.update(loss_nll.item(), x.size(0))

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=utils.bits_per_dim(x, loss_unsup_meter.avg),
                                     acc=acc_meter.avg)
            progress_bar.update(x.size(0))

    writer.add_scalar("train/loss", loss_meter.avg, epoch)
    writer.add_scalar("train/loss_unsup", loss_unsup_meter.avg, epoch)
    writer.add_scalar("train/loss_nll", loss_nll_meter.avg, epoch)
    writer.add_scalar("train/acc", acc_meter.avg, epoch)
    writer.add_scalar("train/bpd", utils.bits_per_dim(x, loss_unsup_meter.avg), epoch)


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


def test(epoch, net, testloader, device, loss_fn, num_samples, writer):
    net.eval()
    loss_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, y in testloader:
                x = x.to(device)
                y = y.to(device)
                z, sldj = net(x, reverse=False)
                loss = loss_fn(z, y=y, sldj=sldj)
                loss_meter.update(loss.item(), x.size(0))

                preds = loss_fn.prior.classify(z.reshape((len(z), -1)))
                preds = preds.reshape(y.shape)
                acc = (preds == y).float().mean().item()
                acc_meter.update(acc, x.size(0))

                progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=utils.bits_per_dim(x, loss_meter.avg),
                                     acc=acc_meter.avg)
                progress_bar.update(x.size(0))

    writer.add_scalar("test/loss", loss_meter.avg, epoch)
    writer.add_scalar("test/acc", acc_meter.avg, epoch)
    writer.add_scalar("test/bpd", utils.bits_per_dim(x, loss_meter.avg), epoch)


parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                help='path to datasets location (default: None)')
parser.add_argument('--logdir', type=str, default=None, required=True, metavar='PATH',
                help='path to log directory (default: None)')
parser.add_argument('--ckptdir', type=str, default=None, required=True, metavar='PATH',
                help='path to ckpt directory (default: None)')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
#parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
parser.add_argument('--num_samples', default=50, type=int, help='Number of samples at test time')
parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
parser.add_argument('--resume', type=str, default=None, required=False, metavar='PATH',
                help='path to checkpoint to resume from (default: None)')
parser.add_argument('--weight_decay', default=5e-5, type=float,
                    help='L2 regularization (only applied to the weight norm scale factors)')

# PAVEL
parser.add_argument('--means', 
                    choices=['from_data', 'pixel_const', 'split_dims', 'split_dims_v2', 'random'], 
                    default='split_dims')
parser.add_argument('--means_r', default=1., type=float,
                    help='r constant used when defyning means')
parser.add_argument('--cov_std', default=1., type=float,
                    help='covariance std for the latent distribution')
parser.add_argument('--save_freq', default=25, type=int, 
                    help='frequency of saving ckpts')
parser.add_argument('--means_trainable', action='store_true', help='Use trainable means')
parser.add_argument('--optimizer', choices=['SGD', 'Adam'], default='Adam')
parser.add_argument('--schedule', choices=['wilson', 'no'], default='no')


args = parser.parse_args()

os.makedirs(args.ckptdir, exist_ok=True)
with open(os.path.join(args.ckptdir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

writer = SummaryWriter(log_dir=args.logdir)

device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
start_epoch = 0

# Note: No normalization applied, since RealNVP expects inputs in (0, 1).
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainloader, testloader, _ = make_sup_data_loaders(
        "cifar10", 
        args.data_path, 
        args.batch_size, 
        args.num_workers, 
        transform_train, 
        transform_test, 
        use_validation=False, 
        shuffle_train=True)

# Model
print('Building model...')
net = RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net, args.gpu_ids)
    cudnn.benchmark = True #args.benchmark

if args.resume is not None:
    print('Resuming from checkpoint at', args.resume)
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']

#PAVEL: we need to find a good way of placing the means
D = (32 * 32 * 3)
r = args.means_r 
cov_std = torch.ones((10)) * args.cov_std
cov_std = cov_std.to(device)
means = torch.zeros((10, D)).to(device)

if args.resume is not None:
    print("Using the means for ckpt")
    means = checkpoint['means']
elif args.means == "from_data":
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

if args.means_trainable:
    print("Using learnable means")
    means = torch.tensor(means_np, requires_grad=True)

writer.add_image("means", means.reshape((10, 3, 32, 32)))
prior = SSLGaussMixture(means, device=device)
loss_fn = FlowLoss(prior)

#PAVEL: check why do we need this
param_groups = utils.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
if args.means_trainable:
    param_groups.append({'name': 'means', 'params': means})

if args.optimizer == "SGD":
    optimizer = optim.SGD(param_groups, lr=args.lr)
elif args.optimizer == "Adam":
    optimizer = optim.Adam(param_groups, lr=args.lr)

for epoch in range(start_epoch, start_epoch + args.num_epochs):

    if args.schedule == 'wilson':
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)	
        writer.add_scalar("hypers/learning_rate", lr, epoch)

    train(epoch, net, trainloader, device, optimizer, loss_fn, args.max_grad_norm, writer)
    test(epoch, net, testloader, device, loss_fn, args.num_samples, writer)

    # Save checkpoint
    if (epoch % args.save_freq == 0):
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'means': means
        }
        os.makedirs(args.ckptdir, exist_ok=True)
        torch.save(state, os.path.join(args.ckptdir, str(epoch)+'.pt'))

    # Save samples and data
    writer.add_image("means", means.reshape((10, 3, 32, 32)))
    images = []
    for i in range(10):
        images_cls = sample(net, loss_fn.prior, args.num_samples // 10, cls=i, device=device)
        images.append(images_cls)
        writer.add_image("samples/class_"+str(i), images_cls)
    images = torch.cat(images)
    os.makedirs(os.path.join(args.ckptdir, 'samples'), exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=args.num_samples //  10 , padding=2, pad_value=255)
    os.makedirs(args.ckptdir, exist_ok=True)
    torchvision.utils.save_image(images_concat, 
                                os.path.join(args.ckptdir, 'samples/epoch_{}.png'.format(epoch)))
