import torch
import numpy as np


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
                z, _ = net(x, reverse=False)
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
