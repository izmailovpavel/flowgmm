import torch
import numpy as np
from tqdm import tqdm
from .shell_util import AverageMeter
from .optim_util import bits_per_dim


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

def get_means(means_type, num_means=10, shape=(3, 32, 32), r=1, trainloader=None, device=None):

    D = np.prod(shape)
    means = torch.zeros((num_means, D)).to(device)

    if means_type == "from_data":
        print("Computing the means")
        means = get_class_means(net, trainloader)
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
        x = torch.sigmoid(x)

        return x


def test_classifier(epoch, net, testloader, device, loss_fn, writer=None):
    net.eval()
    loss_meter = AverageMeter()
    jaclogdet_meter = AverageMeter()
    acc_meter = AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, y in testloader:
                x = x.to(device)
                y = y.to(device)
                z = net(x)
                sldj = net.module.logdet()
                loss = loss_fn(z, y=y, sldj=sldj)
                loss_meter.update(loss.item(), x.size(0))
                jaclogdet_meter.update(sldj.mean().item(), x.size(0))

                preds = loss_fn.prior.classify(z.reshape((len(z), -1)))
                preds = preds.reshape(y.shape)
                acc = (preds == y).float().mean().item()
                acc_meter.update(acc, x.size(0))

                progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=bits_per_dim(x, loss_meter.avg),
                                     acc=acc_meter.avg)
                progress_bar.update(x.size(0))
    if writer is not None:
        writer.add_scalar("test/loss", loss_meter.avg, epoch)
        writer.add_scalar("test/acc", acc_meter.avg, epoch)
        writer.add_scalar("test/bpd", bits_per_dim(x, loss_meter.avg), epoch)
        writer.add_scalar("test/jaclogdet", jaclogdet_meter.avg, epoch)
