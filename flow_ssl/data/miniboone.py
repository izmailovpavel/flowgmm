import numpy as np
import matplotlib.pyplot as plt
import os.path
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from collections import Counter
from sklearn.model_selection import train_test_split
from oil.utils.utils import Expression,export,Named
import pickle
import subprocess

class MINIBOONE(Dataset,metaclass=Named):
    num_classes = 2
    class_weights=None
    ignored_index=-100
    stratify=True
    def __init__(self,root='~/datasets/UCI/miniboone/',train=True, remake=False):
        super().__init__()
        root = os.path.expanduser(root)
        if os.path.exists(root+'dataset.pickle') and not remake:
            with open(root+'dataset.pickle','rb') as f:
                self.__dict__ = pickle.load(f).__dict__
        else:
            if not os.path.exists(root+'MiniBooNE_PID.txt'):
                os.makedirs(root,exist_ok=True)
                subprocess.call("wget http://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt",shell=True)
                subprocess.call(f'cp MiniBooNE_PID.txt {root}',shell=True)
            X,Y = load_data_normalised(root+'MiniBooNE_PID.txt')
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=.1,stratify=Y)
            with open(root+'dataset.pickle','wb') as f:
                pickle.dump(self,f)
        
        self.X = torch.from_numpy(self.X_train if train else self.X_test).float()
        self.Y = torch.from_numpy(self.y_train if train else self.y_test).long()
        self.dim = self.X.shape[1]
    def __getitem__(self,idx):
        return self.X[idx],self.Y[idx]
    def __len__(self):
        return self.X.shape[0]

    # def show_histograms(self, split, vars):

    #     data_split = getattr(self, split, None)
    #     if data_split is None:
    #         raise ValueError('Invalid data split')

    #     util.plot_hist_marginals(data_split.x[:, vars])
    #     plt.show()


def load_data(root_path):
    # NOTE: To remember how the pre-processing was done.
    #print("got here")
    data = pd.read_csv(root_path, names=[str(x) for x in range(50)], delim_whitespace=True)
    nsignal = int(data.iloc[0][0])
    nbackground = int(data.iloc[0][1])
    print("{} signal, {} background".format(nsignal, nbackground))
    minimum = min(nsignal,nbackground)
    labels = np.concatenate((np.ones(minimum),np.zeros(minimum)))
    data = data.iloc[1:].values
    data = np.concatenate((data[:minimum],data[nsignal:nsignal+minimum]))
    #print("got here")
    # Remove some random outliers
    # indices = (data[:, 0] < -100)
    # data = data[~indices]
    # labels = labels[~indices]
    i = 0
    # Remove any features that have too many re-occuring real values.
    # features_to_remove = []
    # for feature in data.T:
    #     c = Counter(feature)
    #     max_count = np.array([v for k, v in sorted(c.items())])[0]
    #     if max_count > 5:
    #         features_to_remove.append(i)
    #     i += 1
    # print(features_to_remove)
    # print(np.array([i for i in range(data.shape[1]) if i not in features_to_remove]))
    # print(data.shape)
    # data = data[:, np.array([i for i in range(data.shape[1]) if i not in features_to_remove])]
    # np.save("~/data/miniboone/data.npy", data)
    return data, labels


def load_data_normalised(root_path):
    data,labels = load_data(root_path)
    data =(data-data.mean(axis=0))/data.std(axis=0)
    return data,labels
