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
from torch import nn

from flow_ssl.realnvp import RealNVP, RealNVPLoss
from tqdm import tqdm
from flow_ssl.distributions import SSLGaussMixture

from tensorboardX import SummaryWriter


def schedule(epoch):
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


def train(epoch, net, classifier, trainloader, device, optimizer, max_grad_norm, writer):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = utils.AverageMeter()
    loss_nll_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            z, sldj = net(x, reverse=False)

            logits = classifier(z.reshape((len(z), -1)))
            loss_nll = F.cross_entropy(logits, y)
            loss = loss_nll

            loss.backward()
            utils.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean().item()

            acc_meter.update(acc, x.size(0))
            loss_meter.update(loss.item(), x.size(0))
            loss_nll_meter.update(loss_nll.item(), x.size(0))

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     acc=acc_meter.avg)
            progress_bar.update(x.size(0))
    writer.add_scalar("train/loss", loss_meter.avg, epoch)
    writer.add_scalar("train/loss_nll", loss_nll_meter.avg, epoch)
    writer.add_scalar("train/acc", acc_meter.avg, epoch)


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


def test(epoch, net, classifier, testloader, device, num_samples, writer):
    net.eval()
    loss_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, y in testloader:
                x = x.to(device)
                y = y.to(device)
                z, sldj = net(x, reverse=False)
                logits = classifier(z.reshape((len(z), -1)))
                loss_nll = F.cross_entropy(logits, y)
                loss = loss_nll

                preds = torch.argmax(logits, dim=1)
                acc = (preds == y).float().mean().item()
                acc_meter.update(acc, x.size(0))
                acc_meter.update(acc, x.size(0))

                progress_bar.set_postfix(loss=loss_meter.avg,
                                     acc=acc_meter.avg)
                progress_bar.update(x.size(0))

    writer.add_scalar("test/loss", loss_meter.avg, epoch)
    writer.add_scalar("test/acc", acc_meter.avg, epoch)


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
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--weight_decay', default=5e-5, type=float,
                    help='L2 regularization (only applied to the weight norm scale factors)')
parser.add_argument('--save_freq', default=25, type=int, 
                    help='frequency of saving ckpts')

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

trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Model
D = (32 * 32 * 3)

print('Building model...')
net = RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)
net = net.to(device)
classifier = nn.Linear(D, 10).to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net, args.gpu_ids)
    cudnn.benchmark = True #args.benchmark

if args.resume:
    # Load checkpoint.
    print('Resuming from checkpoint at ckpts/best.pth.tar...')
    assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('ckpts/best.pth.tar')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['test_loss']
    start_epoch = checkpoint['epoch']


#PAVEL: check why do we need this
param_groups = utils.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
param_groups.append({'name': 'classifier', 'params': classifier.parameters()})

optimizer = optim.SGD(param_groups, lr=args.lr)

for epoch in range(start_epoch, start_epoch + args.num_epochs):
        
    lr = schedule(epoch)
    utils.adjust_learning_rate(optimizer, lr)   
    writer.add_scalar("hypers/learning_rate", lr, epoch)

    train(epoch, net, classifier, trainloader, device, optimizer, args.max_grad_norm, writer)
    test(epoch, net, classifier, testloader, device, args.num_samples, writer)

    # Save checkpoint
    if (epoch % args.save_freq == 0):
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'classifier': classifier.state_dict(),
            'epoch': epoch,
        }
        os.makedirs(args.ckptdir, exist_ok=True)
        torch.save(state, os.path.join(args.ckptdir, str(epoch)+'.pt'))
