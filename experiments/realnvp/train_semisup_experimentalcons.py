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
from PIL import Image

from flow_ssl.realnvp import RealNVP 
from flow_ssl import FlowLoss
from flow_ssl.distributions import SSLGaussMixture
from flow_ssl.data import make_ssl_data_loaders
from flow_ssl.data import NO_LABEL
from flow_ssl.data import TransformTwice


#PAVEL: move to utils later
def linear_rampup(final_value, epoch, num_epochs, start_epoch=0):
    t = (epoch - start_epoch + 1) / num_epochs
    if t > 1:
        t = 1.
    return t * final_value


#PAVEL: think of a good way to reuse the training code for (semi/un/)supervised
def train(epoch, net, trainloader, device, optimizer, loss_fn, 
          label_weight, max_grad_norm, consistency_weight,
          writer, use_unlab=True, 
          ):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = utils.AverageMeter()
    loss_unsup_meter = utils.AverageMeter()
    loss_nll_meter = utils.AverageMeter()
    loss_consistency_meter = utils.AverageMeter()
    jaclogdet_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()
    with tqdm(total=trainloader.batch_sampler.num_labeled) as progress_bar:
        for x, y in trainloader:

            x = x.to(device)
            y = y.to(device)

            labeled_mask = (y != NO_LABEL)

            optimizer.zero_grad()

            with torch.no_grad():
                x = x.to(device)
                z, _ = net(x, reverse=False)
                z = z.detach()

            #PAVEL: simultaneously applying transformation to x and z
            xs_transformed, zs_transformed = [], []
            flip_mask = np.random.choice(2, size=len(x))
            pil_img = Image.fromarray(np.zeros((36, 36, 3)).astype("uint8"))
            for flip, x_img, z_img in zip(flip_mask, x, z):
                assert x_img.shape == torch.Size([3, 32, 32])
                assert z_img.shape == torch.Size([3, 32, 32])
                if flip:
                    x_img = torch.flip(x_img, dims=(2,))
                    z_img = torch.flip(z_img, dims=(2,))
                x_np = x_img.cpu().numpy()
                z_np = z_img.cpu().numpy()
                x_np = np.pad(x_np, ((0, 0), (4, 4), (4, 4)), "constant")
                z_np = np.pad(z_np, ((0, 0), (4, 4), (4, 4)), "constant")
                i, j, h, w = transforms.RandomCrop.get_params(pil_img, output_size=(32, 32))
                x_np = x_np[:, i:i+h, j:j+w]
                z_np = z_np[:, i:i+h, j:j+w]
                xs_transformed.append(torch.tensor(x_np)[None, :])
                zs_transformed.append(torch.tensor(z_np)[None, :])
            t_x = torch.cat(xs_transformed).cuda()
            t_z = torch.cat(zs_transformed).cuda()

            z_t, sldj = net(t_x, reverse=False)

            z_labeled = z_t.reshape((len(z_t), -1))
            z_labeled = z_labeled[labeled_mask]
            y_labeled = y[labeled_mask]

            logits = loss_fn.prior.class_logits(z_labeled)
            loss_nll = F.cross_entropy(logits, y_labeled)

            if use_unlab:
                loss_unsup = loss_fn(z_t, sldj=sldj)
                loss = loss_nll * label_weight + loss_unsup
            else:
                loss_unsup = torch.tensor([0.])
                loss = loss_nll

            loss_consistency = torch.sum((z_t - t_z)**2, dim=[1,2,3])
            loss_consistency = loss_consistency.mean(dim=0)

            loss = loss + loss_consistency * consistency_weight

            loss.backward()
            utils.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_labeled).float().mean().item()

            acc_meter.update(acc, x.size(0))
            loss_meter.update(loss.item(), x.size(0))
            loss_unsup_meter.update(loss_unsup.item(), x.size(0))
            loss_nll_meter.update(loss_nll.item(), x.size(0))
            jaclogdet_meter.update(sldj.mean().item(), x.size(0))
            loss_consistency_meter.update(loss_consistency.item(), x.size(0))

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=utils.bits_per_dim(x, loss_unsup_meter.avg),
                                     acc=acc_meter.avg)
            progress_bar.update(y_labeled.size(0))

    writer.add_image("data/x", x[:10])
    writer.add_image("data/t_x", t_x[:10])
    writer.add_image("latent/z", z[:10])
    writer.add_image("latent/t_z", t_z[:10])
    writer.add_image("latent/z_t", z_t[:10])

    writer.add_scalar("train/loss", loss_meter.avg, epoch)
    writer.add_scalar("train/loss_unsup", loss_unsup_meter.avg, epoch)
    writer.add_scalar("train/loss_nll", loss_nll_meter.avg, epoch)
    writer.add_scalar("train/jaclogdet", jaclogdet_meter.avg, epoch)
    writer.add_scalar("train/acc", acc_meter.avg, epoch)
    writer.add_scalar("train/bpd", utils.bits_per_dim(x, loss_unsup_meter.avg), epoch)
    writer.add_scalar("train/loss_consistency", loss_consistency_meter.avg, epoch)


parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                help='path to datasets location (default: None)')
parser.add_argument('--label_path', type=str, default=None, required=True, metavar='PATH',
                help='path to label txt files location (default: None)')
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
                    default='random')
parser.add_argument('--means_r', default=1., type=float,
                    help='r constant used when defyning means')
parser.add_argument('--cov_std', default=1., type=float,
                    help='covariance std for the latent distribution')
parser.add_argument('--save_freq', default=25, type=int, 
                    help='frequency of saving ckpts')
parser.add_argument('--means_trainable', action='store_true', help='Use trainable means')
parser.add_argument('--optimizer', choices=['SGD', 'Adam'], default='Adam')
parser.add_argument('--schedule', choices=['wilson', 'no'], default='no')
parser.add_argument('--label_weight', default=1., type=float,
                    help='weight of the cross-entropy loss term')
parser.add_argument('--supervised_only', action='store_true', help='Train on labeled data only')
parser.add_argument('--consistency_weight', default=100., type=float,
                    help='weight of the consistency loss term')
parser.add_argument('--eval_freq', default=1, type=int, help='Number of epochs between evaluation')
parser.add_argument('--consistency_rampup', default=1, type=int, help='Number of epochs for consistency loss rampup')


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
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainloader, testloader, _ = make_ssl_data_loaders(
        args.data_path, 
        args.label_path, 
        args.batch_size // 2, 
        args.batch_size // 2, 
        args.num_workers, 
        transform_train, 
        transform_test, 
        use_validation=False)

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
means = utils.get_means(args.means, r=args.means_r, trainloader=trainloader, device=device)

if args.resume is not None:
    print("Using the means for ckpt")
    means = checkpoint['means']


print("Means:", means)
print("Cov std:", cov_std)
means_np = means.cpu().numpy()
#PAVEL: making means invariant to horizontal flips
means_np = means_np.reshape((10, 3, 32, 32))
old_norm = np.linalg.norm(means_np)
means_np = means_np + np.flip(means_np, axis=3)
means_np /= np.linalg.norm(means_np)
means_np *= old_norm
print("Means are invariant to flips:", np.all(means_np == np.flip(means_np, axis=3)))
means_np = means_np.reshape((10, -1))


print("Pairwise dists:", cdist(means_np, means_np))

if args.means_trainable:
    print("Using learnable means")
    means = torch.tensor(means_np, requires_grad=True)
else:
    means = torch.tensor(means_np, requires_grad=False)

writer.add_image("means/means", means.reshape((10, 3, 32, 32)))
writer.add_image("means/flipped", means.reshape((10, 3, 32, 32)).flip(2))
prior = SSLGaussMixture(means, device=device)
loss_fn = FlowLoss(prior)

param_groups = utils.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
if args.means_trainable:
    param_groups.append({'name': 'means', 'params': means})

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
    
    cons_weight = linear_rampup(args.consistency_weight, epoch, args.consistency_rampup, start_epoch)
    
    writer.add_scalar("hypers/learning_rate", lr, epoch)
    writer.add_scalar("hypers/consistency_weight", cons_weight, epoch)

    train(epoch, net, trainloader, device, optimizer, loss_fn, 
          args.label_weight, args.max_grad_norm, cons_weight,
          writer, use_unlab=not args.supervised_only, 
         )

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
    if epoch % args.eval_freq == 0:
        utils.test_classifier(epoch, net, testloader, device, loss_fn, writer)
        writer.add_image("means", means.reshape((10, 3, 32, 32)))
        images = []
        for i in range(10):
            images_cls = utils.sample(net, loss_fn.prior, args.num_samples // 10, cls=i, device=device)
            images.append(images_cls)
            writer.add_image("samples/class_"+str(i), images_cls)
        images = torch.cat(images)
        os.makedirs(os.path.join(args.ckptdir, 'samples'), exist_ok=True)
        images_concat = torchvision.utils.make_grid(images, nrow=args.num_samples //  10 , padding=2, pad_value=255)
        os.makedirs(args.ckptdir, exist_ok=True)
        torchvision.utils.save_image(images_concat, 
                                    os.path.join(args.ckptdir, 'samples/epoch_{}.png'.format(epoch)))
