import os.path
import numpy as np
import torch
from torch.utils.data import Dataset
import subprocess

GDRIVE_DOWNLOAD = lambda FID,FNAME: f"wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget\
        --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
        'https://docs.google.com/uc?export=download&id={FID}' -O- | \
        sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={FID}\" -O {FNAME} && rm -rf /tmp/cookies.txt"

class AG_News(Dataset):
    num_classes=4
    class_weights=None
    ignored_index=-100
    dim = 768
    train_fid,train_fname = '1k40hM-x91OPUwGXhWVU7NoCUOXXEuFR4','ag_news_train.npz'
    test_fid,test_fname = '1cPzyv6UW6Cau87fDj3hxe523UL4qw4EJ','ag_news_test.npz'
    def __init__(self, root=os.path.expanduser('~/datasets/AGNEWS/'), train=True, transform=None, target_transform=None):
        if not os.path.exists(root+self.train_fname):
            os.makedirs(root,exist_ok=True)
            subprocess.call(GDRIVE_DOWNLOAD(self.train_fid,self.train_fname),shell=True)
            subprocess.call(GDRIVE_DOWNLOAD(self.test_fid,self.test_fname),shell=True)
            subprocess.call(f'cp {self.train_fname} {root}',shell=True)
            subprocess.call(f'cp {self.test_fname} {root}',shell=True)

        if transform is not None:
            raise ValueError("Transform should be `None`")
        if train:
            path = os.path.join(root, self.train_fname) 
        else:
            path = os.path.join(root, self.test_fname) 

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
    train_fid,train_fname = '1nkf9k3Cqfxxpk05c0yN0HVVtijrVWjde','yahoo_train.npz'
    test_fid,test_fname = '1Z20AEPvX_mVle_1SFCyrWYBiglmZ6_wv','yahoo_test.npz'
    def __init__(self, root=os.path.expanduser('~/datasets/YAHOO/'), train=True):
        super().__init__()
        if not os.path.exists(root+self.train_fname):
            os.makedirs(root,exist_ok=True)
            subprocess.call(GDRIVE_DOWNLOAD(self.train_fid,self.train_fname),shell=True)
            subprocess.call(GDRIVE_DOWNLOAD(self.test_fid,self.test_fname),shell=True)
            subprocess.call(f'cp {self.train_fname} {root}',shell=True)
            subprocess.call(f'cp {self.test_fname} {root}',shell=True)
        train_path = os.path.join(root, self.train_fname) 
        test_path = os.path.join(root, self.test_fname)
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

        


