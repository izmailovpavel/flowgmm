import torch
import numpy as np
from tqdm import tqdm
from .shell_util import AverageMeter
from .optim_util import bits_per_dim
import torchvision
from flow_ssl.data import NO_LABEL
import sklearn
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


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


def get_class_means_latent(net, trainloader, shape, scale=1.):
    ''' use labeled latent representations to compute means '''
    with torch.no_grad():
        means = torch.zeros(shape)
        n_batches = 0
        with tqdm(total=len(trainloader.dataset)) as progress_bar:
            n_batches = len(trainloader)
            for (x, x2), y_ in trainloader:
                if len(y_.shape) == 2:
                    y, _ = y_[:, 0], y_[:, 1]
                else:
                    y = y_

                z = net(x)
                for i in range(10):
                    means[i] += z[y == i].reshape((-1,) + means[i].shape).sum(dim=0).cpu()
                    #PAVEL: not the right way of computing means
                progress_bar.set_postfix(max_mean=torch.max(means), 
                                         min_mean=torch.min(means))
                progress_bar.update(x.size(0))

        for i in range(10):
            means[i] /= sum(trainloader.dataset.train_labels[:, 0] == i)

        return means*scale


def get_random_data(net, trainloader, shape, num_means):
    with torch.no_grad():
        x, y = next(iter(trainloader))
        if type(x) in [tuple, list]:
            x = x[0]
        z = net(x)
        idx = np.random.randint(x.shape[0], size=num_means)
        means = z[idx]
        classes = np.unique(y.cpu().numpy())
        for cls in classes:
            if cls == NO_LABEL:
                continue
            means[cls] = z[y==cls][0]
        return means


def get_class_means_data(trainloader, shape, scale=1.):
    ''' use labeled data to compute means '''
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


def get_class_means_z(net, trainloader, shape, scale=1.):
    ''' compute latent representation of means in data space '''
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

        z_means = net(means)

        return z_means*scale


def get_means(means_type, num_means=10, shape=(3, 32, 32), r=1, trainloader=None, device=None, net=None):

    D = np.prod(shape)
    means = torch.zeros((num_means, D)).to(device)

    if means_type == "from_data":
        print("Computing the means")
        means = get_class_means_data(trainloader, (num_means, *shape), scale=r)
        means = means.reshape((10, -1)).to(device)

    elif means_type == "from_latent":
        print("Computing the means")
        means = get_class_means_latent(net, trainloader, (num_means, *shape), scale=r)
        means = means.reshape((10, -1)).to(device)

    elif means_type == "from_z":
        print("Computing the means")
        means = get_class_means_z(net, trainloader, (num_means, *shape), scale=r)
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

    elif means_type == "random_data":
        means = get_random_data(net, trainloader, shape, num_means)

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
                    show_classification_images=False, confusion=False):
    net.eval()
    loss_meter = AverageMeter()
    jaclogdet_meter = AverageMeter()
    acc_meter = AverageMeter()
    all_pred_labels = []
    all_xs = []
    all_ys = []
    all_zs = []
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, y in testloader:
                all_xs.append(x.data.numpy())
                all_ys.append(y.data.numpy())
                x = x.to(device)
                y = y.to(device)
                z = net(x)
                all_zs.append(z.cpu().data.numpy())
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
    all_zs = np.vstack(all_zs)
    all_ys = np.hstack(all_ys)

    if writer is not None:
        writer.add_scalar("test/loss{}".format(postfix), loss_meter.avg, epoch)
        writer.add_scalar("test/acc{}".format(postfix), acc_meter.avg, epoch)
        writer.add_scalar("test/bpd{}".format(postfix), bits_per_dim(x, loss_meter.avg), epoch)
        writer.add_scalar("test/jaclogdet{}".format(postfix), jaclogdet_meter.avg, epoch)

        for cls in range(np.max(all_pred_labels)+1):
            num_imgs_cls = (all_pred_labels==cls).sum()
            writer.add_scalar("test_clustering/num_class_{}_{}".format(cls,postfix), 
                    num_imgs_cls, epoch)
            if num_imgs_cls == 0:
                writer.add_scalar("test_clustering/num_class_{}_{}".format(cls,postfix), 
                    0., epoch)
                continue
            writer.add_histogram('label_distributions/num_class_{}_{}'.format(cls,postfix), 
                    all_ys[all_pred_labels==cls], epoch)

            writer.add_histogram(
                'distance_distributions/num_class_{}'.format(cls),
                torch.norm(torch.tensor(all_zs[all_pred_labels==cls]) - loss_fn.prior.means[cls].cpu(), p=2, dim=1),
                epoch
            )

            if show_classification_images:
                images_cls = all_xs[all_pred_labels==cls][:10]
                images_cls = torch.from_numpy(images_cls).float()
                images_cls_concat = torchvision.utils.make_grid(
                        images_cls, nrow=2, padding=2, pad_value=255)
                writer.add_image("test_clustering/class_{}".format(cls), 
                        images_cls_concat)

        if confusion:
            fig = plt.figure(figsize=(8, 8))
            cm = confusion_matrix(all_ys, all_pred_labels)
            cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
            sns.heatmap(cm, annot=True, cmap=plt.cm.Blues)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            conf_img = torch.tensor(np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep=''))
            conf_img = torch.tensor(conf_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))).transpose(0, 2).transpose(1, 2)
            writer.add_image("confusion", conf_img, epoch)
