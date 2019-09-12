import os.path
import numpy as np
import torch
from torch.utils.data import Dataset


class AG_News(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, 
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

        


