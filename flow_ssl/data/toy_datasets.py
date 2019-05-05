import numpy as np
from sklearn import datasets
from PIL import Image


def make_circles_ssl():
    np.random.seed(0)
    n_samples = 1000
    data = datasets.make_circles(n_samples=n_samples, noise=.05, factor=0.4)[0].astype(np.float32)

    labels = np.ones((n_samples,)) * (-1)

    idx1 = [0, 1, 3, 4]
    labels[idx1] = 1
    idx0 = [2, 6, 7, 10, 17, 19, 34]
    labels[idx0] = 0
    
    return data, labels
    
def make_moons_ssl():

    np.random.seed(0)
    n_samples = 1000
    data = datasets.make_moons(n_samples=n_samples, noise=.05, )[0].astype(np.float32)

    labels = np.ones((n_samples,)) * (-1)

    idx1 = [0, 1, 3, 4, 5]#, 9, 11, 14, 16]
    labels[idx1] = 1
    idx0 = [2, 6, 7, 10, 17]#, 19, 34, 13, 15]
    labels[idx0] = 0

    return data, labels

#PAVEL: adapted from ffjord code
def make_dataset_from_img(img_path):
    img = np.array(Image.open(img_path).convert('L'))
    h, w = img.shape
    xx = np.linspace(-4, 4, w)
    yy = np.linspace(-4, 4, h)
    xx, yy = np.meshgrid(xx, yy)
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    
    means = np.concatenate([xx, yy], 1)
    img = img.max() - img
    probs = img.reshape(-1) / img.sum()
    std = np.array([8 / w / 2, 8 / h / 2])
    
    def sample_data(data=None, rng=None, batch_size=200):
        """data and rng are ignored."""
        inds = np.random.choice(int(probs.shape[0]), int(batch_size), p=probs)
        m = means[inds]
        samples = np.random.randn(*m.shape) * std + m
        return samples
    
    X = sample_data(data="dummy", batch_size=1000)
    X[:, 1] *=  -1.
    return X.astype(np.float32), np.ones(1000) * (-1)

def make_dataset_from_npz(npz_path):
    f = np.load(npz_path)
    data = f["data"].astype(np.float32)
    labels = f["labels"].astype(np.int)
    return data, labels
 
