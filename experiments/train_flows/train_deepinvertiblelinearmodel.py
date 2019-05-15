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
from tensorboardX import SummaryWriter
from tqdm import tqdm

from flow_ssl.realnvp import RealNVP
from flow_ssl.data import make_sup_data_loaders


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

parser.add_argument('--save_freq', default=25, type=int, 
                    help='frequency of saving ckpts')
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
        "CIFAR10", 
        args.data_path, 
        args.batch_size, 
        args.num_workers, 
        transform_train, 
        transform_test, 
        use_validation=False, 
        shuffle_train=True)

# Model
D = (32 * 32 * 3)

print('Building model...')
net = RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)
net = net.to(device)
classifier = nn.Linear(D, 10).to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net, args.gpu_ids)
    cudnn.benchmark = True #args.benchmark

if args.resume is not None:
    print('Resuming from checkpoint at', args.resume)
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    classifier.load_state_dict(checkpoint['classifier'])
    start_epoch = checkpoint['epoch']


#PAVEL: check why do we need this
param_groups = utils.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
param_groups.append({'name': 'classifier', 'params': classifier.parameters()})

if args.optimizer == "SGD":
    optimizer = optim.SGD(param_groups, lr=args.lr)
elif args.optimizer == "Adam":
    optimizer = optim.Adam(param_groups, lr=args.lr)

for epoch in range(start_epoch, args.num_epochs):

    if args.schedule == 'wilson':
        lr = utils.wilson_schedule(args.lr, epoch, args.num_epochs)
        utils.adjust_learning_rate(optimizer, lr)	
    else:
        lr = args.lr

    writer.add_scalar("hypers/learning_rate", lr, epoch)


    train(epoch, net, classifier, trainloader, device, optimizer, args.max_grad_norm, writer)
    utils.test_classifier(epoch, net, classifier, testloader, device, writer)

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
