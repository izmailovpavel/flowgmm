import torchvision
from torchvision.datasets import SVHN

class SVHN_(SVHN):
    """
    Reimplementation of SVHN datasets with the same interface as MNIST.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, 
                 download=False):
               
        super(SVHN_, self).__init__(root, split="train" if train else "test", 
                         transform=transform, target_transform=target_transform,
                         download=download)
        self.train = train
        if train:
            self.train_data = self.data
            self.train_labels = self.labels
        else:
            self.test_data = self.data
            self.test_labels = self.labels
        delattr(self, "data")
        delattr(self, "labels")
        
    def __getattr__(self, attr):
        if attr == "data":
            if self.train:
                return self.train_data
            else:
                return self.test_data
        elif attr == "labels":
            if self.train:
                return self.train_labels
            else:
                return self.test_labels
        else:
            raise AttributeError(attr)
