import torch
import numpy as np
from tqdm import tqdm
from .shell_util import AverageMeter
from .optim_util import bits_per_dim
import torchvision


def wilson_schedule(lr_init, epoch, num_epochs):
    lr_ratio = 0.01
    t = epoch / num_epochs
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor


def get_class_means(net, trainloader, shape):
    with torch.no_grad():
        means = torch.zeros(shape)
        n_batches = 0
        with tqdm(total=len(trainloader.dataset)) as progress_bar:
            n_batches = len(trainloader)
            for x, y in trainloader:
                z, _ = net(x)
                for i in range(10):
                    means[i] += z[y == i].sum(dim=0).cpu() 
                    #PAVEL: not the right way of computing means
                progress_bar.set_postfix(max_mean=torch.max(means), 
                                         min_mean=torch.min(means))
                progress_bar.update(x.size(0))
        means /= 5000
        return means


def get_class_means_unlabeled(trainloader, shape, scale=500):
    with torch.no_grad():
        means = torch.zeros(shape)
        with tqdm(total=len(trainloader.dataset)) as progress_bar:
            for (x, x2), y_ in trainloader:
                if len(y_.shape) == 2:
                    y, _ = y_[:, 0], y_[:, 1]
                else:
                    y = y_

                for i in range(10):
                    means[i] += x[y == i].sum(dim=0).cpu()

        for i in range(10):
            means[i] /= sum(trainloader.dataset.train_labels[:, 0] == i)

        return means*scale


def get_means(means_type, num_means=10, shape=(3, 32, 32), r=1, trainloader=None, device=None):

    D = np.prod(shape)
    means = torch.zeros((num_means, D)).to(device)

    if means_type == "from_data":
        print("Computing the means")
        means = get_class_means_unlabeled(trainloader, (num_means, *shape))
        means = means.reshape((10, -1)).to(device)

    elif means_type == "pixel_const":
        for i in range(num_means):
            means[i, :] = r * (i-4)
    
    elif means_type == "split_dims":
        mean_portion = D // num_means
        for i in range(num_means):
            means[i, i*mean_portion:(i+1)*mean_portion] = r

    elif means_type == "random":
        for i in range(num_means):
            means[i] = r * torch.randn(D)
    else:
        raise NotImplementedError(means_type)

    return means


def sample(net, prior, batch_size, cls, device, sample_shape):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    with torch.no_grad():
        if cls is not None:
            z = prior.sample((batch_size,), gaussian_id=cls)
        else:
            z = prior.sample((batch_size,))
        x = net.module.inverse(z)

        return x


def test_classifier(epoch, net, testloader, device, loss_fn, writer=None, postfix="",
                    show_classification_images=False):
    net.eval()
    loss_meter = AverageMeter()
    jaclogdet_meter = AverageMeter()
    acc_meter = AverageMeter()
    all_pred_labels = []
    all_xs = []
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, y in testloader:
                all_xs.append(x.data.numpy())
                x = x.to(device)
                y = y.to(device)
                z = net(x)
                sldj = net.module.logdet()
                loss = loss_fn(z, y=y, sldj=sldj)
                loss_meter.update(loss.item(), x.size(0))
                jaclogdet_meter.update(sldj.mean().item(), x.size(0))

                preds = loss_fn.prior.classify(z.reshape((len(z), -1)))
                preds = preds.reshape(y.shape)
                all_pred_labels.append(preds.cpu().data.numpy())
                acc = (preds == y).float().mean().item()
                acc_meter.update(acc, x.size(0))

                progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=bits_per_dim(x, loss_meter.avg),
                                     acc=acc_meter.avg)
                progress_bar.update(x.size(0))
    all_pred_labels = np.hstack(all_pred_labels)
    all_xs = np.vstack(all_xs)
    
    if writer is not None:
        writer.add_scalar("test/loss{}".format(postfix), loss_meter.avg, epoch)
        writer.add_scalar("test/acc{}".format(postfix), acc_meter.avg, epoch)
        writer.add_scalar("test/bpd{}".format(postfix), bits_per_dim(x, loss_meter.avg), epoch)
        writer.add_scalar("test/jaclogdet{}".format(postfix), jaclogdet_meter.avg, epoch)

        for cls in range(np.max(all_pred_labels)):
            num_imgs_cls = (all_pred_labels==cls).sum()
            writer.add_scalar("test_clustering/num_class_{}_{}".format(cls,postfix), 
                    num_imgs_cls, epoch)
            if num_imgs_cls == 0:
                continue
            if show_classification_images:
                images_cls = all_xs[all_pred_labels==cls][:10]
                images_cls = torch.from_numpy(images_cls).float()
                images_cls_concat = torchvision.utils.make_grid(
                        images_cls, nrow=2, padding=2, pad_value=255)
                writer.add_image("test_clustering/class_{}".format(cls), 
                        images_cls_concat)
