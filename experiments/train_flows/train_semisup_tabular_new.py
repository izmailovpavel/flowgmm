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

import flow_ssl
from flow_ssl.realnvp import RealNVPTabular
from flow_ssl import FlowLoss
from flow_ssl.distributions import SSLGaussMixture
from flow_ssl.data import make_ssl_data_loaders
from flow_ssl.data import make_sup_data_loaders
from flow_ssl.data import NO_LABEL


#PAVEL: think of a good way to reuse the training code for (semi/un/)supervised
def train(epoch, net, trainloader, device, optimizer, opt_gmm, loss_fn,
          label_weight, max_grad_norm, 
          writer, use_unlab=True, 
          ):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = utils.AverageMeter()
    loss_unsup_meter = utils.AverageMeter()
    loss_nll_meter = utils.AverageMeter()
    jaclogdet_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()
    dist_loss_meter = utils.AverageMeter()
    with tqdm(total=trainloader.batch_sampler.num_labeled) as progress_bar:
        for x1, y in trainloader:

            x1 = x1.to(device)
            y = y.to(device)

            labeled_mask = (y != NO_LABEL)

            optimizer.zero_grad()

            z1 = net(x1)
            sldj = net.module.logdet()

            z_labeled = z1.reshape((len(z1), -1))
            z_labeled = z_labeled[labeled_mask]
            y_labeled = y[labeled_mask]

            logits_labeled = loss_fn.prior.class_logits(z_labeled)
            loss_nll = F.cross_entropy(logits_labeled, y_labeled)

            if use_unlab:
                loss_unsup = loss_fn(z1, sldj=sldj)
                loss = loss_nll * label_weight + loss_unsup
            else:
                loss_unsup = torch.tensor([0.])
                loss = loss_nll

            #TODO: fix
            sigma = 1 / loss_fn.prior.inv_cov_stds[0]
            means = loss_fn.prior.means
            means_norms = torch.norm(means, dim=1)**2
            dists = means_norms[:, None] + means_norms[None, :] - 2*means@means.t()
            dist_loss = -(dists * (dists < (2*sigma)**2).float()).sum()
            loss += dist_loss

            loss.backward()
            utils.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()
            opt_gmm.step()


            preds = torch.argmax(logits_labeled, dim=1)
            acc = (preds == y_labeled).float().mean().item()

            acc_meter.update(acc, x1.size(0))
            loss_meter.update(loss.item(), x1.size(0))
            dist_loss_meter.update(dist_loss.item(), x1.size(0))
            loss_unsup_meter.update(loss_unsup.item(), x1.size(0))
            loss_nll_meter.update(loss_nll.item(), x1.size(0))
            jaclogdet_meter.update(sldj.mean().item(), x1.size(0))

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=utils.bits_per_dim(x1, loss_unsup_meter.avg),
                                     acc=acc_meter.avg)
            progress_bar.update(y_labeled.size(0))

    writer.add_scalar("train/loss", loss_meter.avg, epoch)
    writer.add_scalar("train/loss_unsup", loss_unsup_meter.avg, epoch)
    writer.add_scalar("train/loss_nll", loss_nll_meter.avg, epoch)
    writer.add_scalar("train/jaclogdet", jaclogdet_meter.avg, epoch)
    writer.add_scalar("train/acc", acc_meter.avg, epoch)
    writer.add_scalar("train/dist_loss", dist_loss_meter.avg, epoch)
    writer.add_scalar("train/bpd", utils.bits_per_dim(x1, loss_unsup_meter.avg), epoch)


parser = argparse.ArgumentParser(description='RealNVP on Tabular Data')

parser.add_argument('--dataset', type=str, default="AG_News", required=True, metavar='DATA',
                help='Dataset name (default: AG_News)')
parser.add_argument('--use_validation', action='store_true', help='Use trainable validation set')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                help='path to datasets location (default: None)')
parser.add_argument('--label_path', type=str, default=None, required=True, metavar='PATH',
                help='path to label txt files location (default: None)')
#parser.add_argument('--flow', type=str, default="RealNVP", required=False, metavar='PATH',
#                help='Flow model to use (default: RealNVP)')
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
                    choices=['from_data', 'pixel_const', 'split_dims', 
                        'split_dims_v2', 'random', 'random_data'], 
                    default='random')
parser.add_argument('--means_r', default=1., type=float,
                    help='r constant used when defyning means')
parser.add_argument('--cov_std', default=1., type=float,
                    help='covariance std for the latent distribution')
parser.add_argument('--save_freq', default=25, type=int, 
                    help='frequency of saving ckpts')
#parser.add_argument('--means_trainable', action='store_true', help='Use trainable means')
parser.add_argument('--optimizer', choices=['SGD', 'Adam'], default='Adam')
parser.add_argument('--schedule', choices=['wilson', 'no'], default='no')
parser.add_argument('--label_weight', default=1., type=float,
                    help='weight of the cross-entropy loss term')
parser.add_argument('--supervised_only', action='store_true', help='Train on labeled data only')
parser.add_argument('--eval_freq', default=1, type=int, help='Number of epochs between evaluation')
parser.add_argument('--consistency_rampup', default=1, type=int, help='Number of epochs for consistency loss rampup')

parser.add_argument('--swa', action='store_true', help='SWA (default: off)')
parser.add_argument('--swa_start', type=float, default=0, metavar='N', help='Steps before SWA start (default: 0)')
parser.add_argument('--swa_freq', type=float, default=10, metavar='N', help='SWA upd freq (default: 10)')
parser.add_argument('--swa_lr', type=float, default=0.001, metavar='LR', help='SWA LR (default: 0.001)')

parser.add_argument('--gmm_trainable', action='store_true')
parser.add_argument('--means_trainable', action='store_true')
parser.add_argument('--weights_trainable', action='store_true')
parser.add_argument('--covs_trainable', action='store_true')
parser.add_argument('--lr_gmm', default=1e-2, type=float)
# parser.add_argument('--means_reg', action='store_true')
# parser.add_argument('--joint', action='store_true')
# parser.add_argument('--flow_iters', default=300, type=int)
# parser.add_argument('--gmm_iters', default=100, type=int)
parser.add_argument('--n_gaussians', default=4, type=int, 
                    help='number of Gaussians')


args = parser.parse_args()

os.makedirs(args.ckptdir, exist_ok=True)
with open(os.path.join(args.ckptdir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

writer = SummaryWriter(log_dir=args.logdir)

device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
start_epoch = 0

# Note: No normalization applied, since RealNVP expects inputs in (0, 1).

trainloader, testloader, _ = make_ssl_data_loaders(
        args.data_path, 
        args.label_path, 
        args.batch_size // 2, 
        args.batch_size // 2, 
        args.num_workers, 
        None, 
        None, 
        use_validation=args.use_validation,
        dataset=args.dataset.lower())

if args.dataset.lower() == "ag_news":
    embed_size = 768
    n_class = 4
    unlabeled_classes = [0]
    labeled_classes = [1,2,3]
#    n_class = 7
#    unlabeled_classes = [0, 1]
#    labeled_classes = [2,3]
#    unlabeled_classes = []
#    labeled_classes = [0,1,2,3]

for cls in unlabeled_classes:
    labels = trainloader.dataset.train_labels 
    labels[labels == cls] = NO_LABEL


labels = trainloader.dataset.train_labels
for cls in range(n_class):
    print("Class {}: {} data".format(cls, (labels==cls).sum()))

if args.swa:
    #raise NotImplementedError("SWA not yet supported")
    bn_loader, _, _ = make_sup_data_loaders(
            args.data_path, 
            args.batch_size, 
            args.num_workers, 
            None, 
            None, 
            use_validation=args.use_validation,
            shuffle_train=True,
            dataset=args.dataset.lower())
    

# Model
print('Building RealNVPTabular model...')
model_cfg = RealNVPTabular
net = model_cfg(num_coupling_layers=10, in_dim=embed_size, num_layers=1, hidden_dim=512)
#net = model_cfg(num_coupling_layers=10, in_dim=embed_size, num_layers=1, hidden_dim=256)
#net = model_cfg(num_coupling_layers=7, in_dim=embed_size, num_layers=1, hidden_dim=256)
#net = model_cfg(num_coupling_layers=5, in_dim=embed_size, num_layers=1, hidden_dim=768)
#net = model_cfg(num_coupling_layers=5, in_dim=embed_size, num_layers=1, hidden_dim=512)
#net = model_cfg(num_coupling_layers=5, in_dim=embed_size, num_layers=1, hidden_dim=256)
#net = model_cfg(num_coupling_layers=4, in_dim=embed_size, num_layers=1, hidden_dim=256)
#net = model_cfg(num_coupling_layers=4, in_dim=embed_size, num_layers=1, hidden_dim=128)
print("Model contains {} parameters".format(sum([p.numel() for p in net.parameters()])))

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
r = args.means_r 
inv_cov_std = torch.ones((args.n_gaussians,), device=device) / args.cov_std
#TODO: need to used cov_std
means = utils.get_means(args.means, r=args.means_r, num_means=args.n_gaussians, trainloader=trainloader, 
                        shape=(embed_size,), device=device, net=net)

if args.resume is not None:
    print("Using the means for ckpt")
    means = checkpoint['means']


means_np = means.cpu().numpy()
print("Pairwise dists:", cdist(means_np, means_np))

if args.gmm_trainable:
    args.means_trainable = True
    args.covs_trainable = True
    args.weights_trainable = True

if args.means_trainable:
    print("Using learnable means")
    means = torch.tensor(means_np, requires_grad=True, device=device)

prior = SSLGaussMixture(means, inv_cov_std, device=device)
prior.weights.requires_grad = args.weights_trainable
prior.inv_cov_stds.requires_grad = args.covs_trainable
loss_fn = FlowLoss(prior)
print("Means:", prior.means)
print("Inv cov std:", prior.inv_cov_stds)

#PAVEL: check why do we need this
param_groups = utils.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
if args.means_trainable:
    param_groups.append({'name': 'means', 'params': means})

if args.optimizer == "SGD":
    opt_gmm = optim.SGD([prior.means, prior.weights, prior.inv_cov_stds],
                               lr=args.lr_gmm, weight_decay=0.)
    optimizer = optim.SGD(param_groups, lr=args.lr)
elif args.optimizer == "Adam":
    optimizer = optim.Adam(param_groups, lr=args.lr)
    #opt_gmm = optim.Adam([
    #        {"name": "means", "params": [prior.means], "lr": args.lr_gmm},
    #        {"name": "gmm_weight_cov", "params": [prior.weights, prior.inv_cov_stds], "lr": args.lr}], 
    #        weight_decay=0.)
    opt_gmm = optim.Adam([
            {"name": "means", "params": [prior.means], "lr": args.lr_gmm},
            {"name": "gmm_weight_cov", "params": [prior.weights, prior.inv_cov_stds], "lr": args.lr_gmm}], 
            weight_decay=0.)
else:
    raise ValueError("Unknown optimizer {}".format(args.optimizer))

if args.swa:
    from torchcontrib.optim import SWA
    optimizer = optim.SGD(param_groups, lr=args.lr)
    optimizer = SWA(optimizer, args.swa_start, args.swa_freq, args.swa_lr)


for epoch in range(start_epoch, args.num_epochs):

    if args.schedule == 'wilson':
        lr = utils.wilson_schedule(args.lr, epoch, args.num_epochs)
        utils.adjust_learning_rate(optimizer, lr)	
        lr_gmm = utils.wilson_schedule(args.lr_gmm, epoch, args.num_epochs)
        utils.adjust_learning_rate(opt_gmm, lr_gmm)
    else:
        lr = args.lr
        lr_gmm = args.lr_gmm
    
    writer.add_scalar("hypers/learning_rate", lr, epoch)
    writer.add_scalar("hypers/learning_rate_gmm", lr_gmm, epoch)

    train(epoch, net, trainloader, device, optimizer, opt_gmm, loss_fn, 
          args.label_weight, args.max_grad_norm, 
          writer, use_unlab=not args.supervised_only, 
         )

    # Save checkpoint
    if (epoch % args.save_freq == 0):
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'means': prior.means,
        }
        os.makedirs(args.ckptdir, exist_ok=True)
        torch.save(state, os.path.join(args.ckptdir, str(epoch)+'.pt'))

    # Save samples and data
    if epoch % args.eval_freq == 0:
        utils.test_classifier(epoch, net, testloader, device, loss_fn, writer)

        weights_softmax = F.softmax(prior.weights)
        for i in range(len(prior.means)):
            writer.add_scalar("train/gaussian_weight/{}".format(i), weights_softmax[i], epoch)

        if args.swa:
            optimizer.swap_swa_sgd() 
            print("updating bn")
            SWA.bn_update(bn_loader, net)
            utils.test_classifier(epoch, net, testloader, device, loss_fn, 
                    writer, postfix="_swa")
        os.makedirs(args.ckptdir, exist_ok=True)

        if args.swa:
            optimizer.swap_swa_sgd() 
