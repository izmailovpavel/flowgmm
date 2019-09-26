import os.path
import numpy as np
import torch
from torch.utils.data import Dataset


class AG_News(Dataset):
    num_classes=4
    class_weights=None
    ignored_index=-100
    dim = 768
    def __init__(self, root=os.path.expanduser('~/datasets/AGNEWS/'), train=True, transform=None, target_transform=None, 
                 download=False):
        if download:
            raise ValueError("Please run the data preparation scripts and set `download=False`")
        if transform is not None:
            raise ValueError("Transform should be `None`")
        if train:
            path = os.path.join(root, "ag_news_train.npz") 
        else:
            path = os.path.join(root, "ag_news_test.npz") 

        data_labels = np.load(path)
        data, labels = data_labels["encodings"], data_labels["labels"]
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)
        if train:
            self.train_data = data
            self.train_labels = labels
        else:
            self.test_data = data
            self.test_labels = labels
        self.train = train

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)

    def __getitem__(self, idx):
        if self.train:
            return self.train_data[idx], self.train_labels[idx]
        else:
            return self.test_data[idx], self.test_labels[idx]



class YAHOO(Dataset):
    num_classes=10
    class_weights=None
    ignored_index=-100
    dim = 768
    def __init__(self, root=os.path.expanduser('~/datasets/YAHOO/'), train=True):
        super().__init__()
        train_path = os.path.join(root, "yahoo_train.npz") 
        test_path = os.path.join(root, "yahoo_test.npz")
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        self.X_train, self.y_train = train_data["encodings"], train_data["labels"]
        self.X_test, self.y_test = test_data["encodings"], test_data["labels"]
        self.X_test = self.X_test[:10000]
        self.y_test = self.y_test[:10000]
        self.X = torch.from_numpy(self.X_train if train else self.X_test).float()
        self.Y = torch.from_numpy(self.y_train if train else self.y_test).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx],self.Y[idx]

        


