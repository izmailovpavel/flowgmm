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

import flow_ssl
from flow_ssl.realnvp import RealNVP
from flow_ssl import FlowLoss
from tqdm import tqdm
from torch import distributions

from tensorboardX import SummaryWriter

from flow_ssl.data import make_sup_data_loaders

def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm, writer):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()
            z = net(x)
            sldj = net.module.logdet()
            loss = loss_fn(z, sldj=sldj)
            loss_meter.update(loss.item(), x.size(0))

            loss.backward()
            utils.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=utils.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))
    writer.add_scalar("train/loss", loss_meter.avg, epoch)
    writer.add_scalar("train/bpd", utils.bits_per_dim(x, loss_meter.avg), epoch)


def test(epoch, net, testloader, device, loss_fn, num_samples, writer):
    net.eval()
    loss_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, _ in testloader:
                x = x.to(device)
                z = net(x)
                sldj = net.module.logdet()
                loss = loss_fn(z, sldj=sldj)
                loss_meter.update(loss.item(), x.size(0))

                progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=utils.bits_per_dim(x, loss_meter.avg))
                progress_bar.update(x.size(0))

    writer.add_scalar("test/loss", loss_meter.avg, epoch)
    writer.add_scalar("test/bpd", utils.bits_per_dim(x, loss_meter.avg), epoch)


parser = argparse.ArgumentParser(description='RealNVP')

parser.add_argument('--dataset', type=str, default="CIFAR10", required=True, metavar='DATA',
                help='Dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                help='path to datasets location (default: None)')
parser.add_argument('--logdir', type=str, default=None, required=True, metavar='PATH',
                help='path to log directory (default: None)')
parser.add_argument('--flow', type=str, default="RealNVP", required=False, metavar='PATH',
                help='Flow model to use (default: RealNVP)')
parser.add_argument('--ckptdir', type=str, default=None, required=True, metavar='PATH',
                help='path to ckpt directory (default: None)')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
parser.add_argument('--num_samples', default=50, type=int, help='Number of samples at test time')
parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--weight_decay', default=5e-5, type=float,
                    help='L2 regularization (only applied to the weight norm scale factors)')
parser.add_argument('--use_validation', action='store_true', help='Use trainable validation set')

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

if args.dataset.lower() == "mnist":
    img_shape = (1, 28, 28)
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor()
    ])
elif args.dataset.lower() in ["cifar10", "svhn"]:
    img_shape = (3, 32, 32)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
else:
    raise ValueError("Unsupported dataset "+args.dataset)

transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainloader, testloader, _ = make_sup_data_loaders(
        args.data_path, 
        args.batch_size, 
        args.num_workers, 
        transform_train, 
        transform_test, 
        use_validation=args.use_validation,
        shuffle_train=True,
        dataset=args.dataset.lower())

# Model
print('Building {} model...'.format(args.flow))
model_cfg = getattr(flow_ssl, args.flow)
net = model_cfg(in_channels=img_shape[0])
print("Model contains {} parameters".format(sum([p.numel() for p in net.parameters()])))

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

D = np.prod(img_shape)
D = int(D)
prior = distributions.MultivariateNormal(torch.zeros(D).to(device),
                                                     torch.eye(D).to(device))
loss_fn = FlowLoss(prior)

#PAVEL: check why do we need this
param_groups = utils.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
optimizer = optim.Adam(param_groups, lr=args.lr)

for epoch in range(start_epoch, start_epoch + args.num_epochs):
    train(epoch, net, trainloader, device, optimizer, loss_fn, args.max_grad_norm, writer)
    test(epoch, net, testloader, device, loss_fn, args.num_samples, writer)

    # Save checkpoint
    if (epoch % args.save_freq == 0):
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }
        os.makedirs(args.ckptdir, exist_ok=True)
        torch.save(state, os.path.join(args.ckptdir, str(epoch)+'.pt'))

    # Save samples and data
    images = utils.sample(net, loss_fn.prior, args.num_samples,
                              cls=None, device=device, sample_shape=img_shape)
    writer.add_image("samples/unsup", images)
    os.makedirs(os.path.join(args.ckptdir, 'samples'), exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=int(args.num_samples ** 0.5), padding=2, pad_value=255)
    os.makedirs(args.ckptdir, exist_ok=True)
    torchvision.utils.save_image(images_concat, 
                                os.path.join(args.ckptdir, 'samples/epoch_{}.png'.format(epoch)))
