"""
Code adapted from https://github.com/chrischute/real-nvp, which is in turn
adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import utils
import numpy as np
from scipy.spatial.distance import cdist

from flow_ssl.realnvp import RealNVP, RealNVPLoss
from tqdm import tqdm
from flow_ssl.distributions import SSLGaussMixture

from tensorboardX import SummaryWriter


def main(args):

    writer = SummaryWriter(log_dir=args.logdir)

    device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
    start_epoch = 0

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
        cudnn.benchmark = True #args.benchmark

    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/best.pth.tar')
        net.load_state_dict(checkpoint['net'])
        global best_loss
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']

    #PAVEL: we need to find a good way of placing the means
    D = (32 * 32 * 3)
    r = 2. 
    means = torch.zeros((10, D)).to(device)
    #mean_portion = D // 10
    for i in range(10):
        #means[i, i*mean_portion:(i+1)*mean_portion] = r
        means[i, :] = r * (i-4)
    
    #print("Computing the means")
    #means = get_class_means(net, trainloader)
    #writer.add_image("means", means)
    #means = means.reshape((10, -1)).to(device)
    print(means)
    means_np = means.cpu().numpy()
    print("Pairwise dists:", cdist(means_np, means_np))

    prior = SSLGaussMixture(means, device=device)
    loss_fn = RealNVPLoss(prior)

    #PAVEL: check why do we need this
    param_groups = utils.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    optimizer = optim.Adam(param_groups, lr=args.lr)

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, net, trainloader, device, optimizer, loss_fn, args.max_grad_norm, writer)
        test(epoch, net, testloader, device, loss_fn, args.num_samples, writer)


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
    acc_meter = utils.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            z, sldj = net(x, reverse=False)
            #print(z.shape)
            #loss = loss_fn(z, y=y, sldj=sldj)
            loss = loss_fn(z, y=y, sldj=sldj)
            #print(loss)
            loss_meter.update(loss.item(), x.size(0))

            preds = loss_fn.prior.classify(z.reshape((len(z), -1)))
            preds = preds.reshape(y.shape)
            acc = (preds == y).float().mean().item()
            acc_meter.update(acc, x.size(0))

            loss.backward()
            utils.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=utils.bits_per_dim(x, loss_meter.avg),
                                     acc=acc_meter.avg)
            progress_bar.update(x.size(0))
    writer.add_scalar("train/loss", loss_meter.avg, epoch)
    writer.add_scalar("train/acc", acc_meter.avg, epoch)
    writer.add_scalar("train/bpd", utils.bits_per_dim(x, loss_meter.avg), epoch)


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
    global best_loss
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

    # Save checkpoint
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/best.pth.tar')
        best_loss = loss_meter.avg

    # Save samples and data
    #PAVEL: make class_conditional sampling
    images = []
    for i in range(10):
        images_cls = sample(net, loss_fn.prior, num_samples // 10, cls=i, device=device)
        images.append(images_cls)
        writer.add_image("samples/class_"+str(i), images_cls)
    images = torch.cat(images)
    os.makedirs('samples', exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=num_samples //  10 , padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

    parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
    parser.add_argument('--logdir', type=str, default=None, required=True, metavar='PATH',
                    help='path to log directory (default: None)')
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

    best_loss = 0

    main(parser.parse_args())
