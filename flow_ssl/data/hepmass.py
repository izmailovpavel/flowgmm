import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from os.path import join
import os.path
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from oil.utils.utils import Expression,export,Named
import subprocess
from sklearn.model_selection import train_test_split

class HEPMASS(Dataset,metaclass=Named):
    """
    The HEPMASS data set.
    http://archive.ics.uci.edu/ml/datasets/HEPMASS
    """
    num_classes = 2
    class_weights=None
    ignored_index=-100
    stratify=True
    def __init__(self,root='~/datasets/UCI/hepmass/',train=True,remake=False):
        super().__init__()
        root = os.path.expanduser(root)
        if os.path.exists(root+'dataset.pickle') and not remake:
            with open(root+'dataset.pickle','rb') as f:
                self.__dict__ = pickle.load(f).__dict__
        else:
            if not os.path.exists(root+'1000_train.csv'):
                os.makedirs(root,exist_ok=True)
                subprocess.call("wget http://archive.ics.uci.edu/ml/machine-learning-databases/00347/1000_train.csv.gz",shell=True)
                subprocess.call("wget http://archive.ics.uci.edu/ml/machine-learning-databases/00347/1000_test.csv.gz",shell=True)
                subprocess.call('gzip -d 1000_train.csv.gz',shell=True)
                subprocess.call('gzip -d 1000_test.csv.gz',shell=True)
                subprocess.call(f'cp 1000_train.csv {root}',shell=True)
                subprocess.call(f'cp 1000_test.csv {root}',shell=True)
            self.trn, self.tst, self.y_trn, self.y_tst = load_data_no_discrete_normalised_as_array(root)
            #self.trn, _, self.y_trn, _ = train_test_split(self.trn, self.y_trn, train_size=140000,stratify=self.y_trn)
            with open(root+'dataset.pickle','wb') as f:
                pickle.dump(self,f)
        self.X = torch.from_numpy(self.trn if train else self.tst).float()
        self.y = torch.from_numpy(self.y_trn if train else self.y_tst).long()
        self.dim = self.X.shape[1]
        print(len(self))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]

    # def show_histograms(self, split, vars):

    #     data_split = getattr(self, split, None)
    #     if data_split is None:
    #         raise ValueError('Invalid data split')

    #     util.plot_hist_marginals(data_split.x[:, vars])
    #     plt.show()


def load_data(path):

    data_train = pd.read_csv(filepath_or_buffer=join(path, "1000_train.csv"), index_col=False).sample(frac=.2)
    data_test = pd.read_csv(filepath_or_buffer=join(path, "1000_test.csv"), index_col=False).sample(frac=.04)

    return data_train, data_test


def load_data_no_discrete(path):
    """
    Loads the positive class examples from the first 10 percent of the dataset.
    """
    data_train, data_test = load_data(path)

    # Gets rid of any background noise examples i.e. class label 0.
    #data_train = data_train[data_train[data_train.columns[0]] == 1]
    train_labels = data_train[data_train.columns[0]]
    data_train = data_train.drop(data_train.columns[0], axis=1)
    #data_test = data_test[data_test[data_test.columns[0]] == 1]
    test_labels = data_test[data_test.columns[0]]
    data_test = data_test.drop(data_test.columns[0], axis=1)
    # Because the data set is messed up!
    data_test = data_test.drop(data_test.columns[-1], axis=1)
    print("Fraction of positives {} train, {} test".format(train_labels.mean(), test_labels.mean()))
    return data_train, data_test, train_labels, test_labels


def load_data_no_discrete_normalised(path):

    data_train, data_test, train_labels, test_labels = load_data_no_discrete(path)
    mu = data_train.mean()
    s = data_train.std()
    data_train = (data_train - mu)/s
    data_test = (data_test - mu)/s

    return data_train, data_test, train_labels, test_labels


def load_data_no_discrete_normalised_as_array(path):

    data_train, data_test, label_train, label_test = load_data_no_discrete_normalised(path)
    data_train, data_test = data_train.values, data_test.values
    y_train = np.array(label_train)
    #print(label_train)
    #print(label_test)
    y_test = np.array(label_test)

    i = 0
    # Remove any features that have too many re-occurring real values.
    features_to_remove = []
    for feature in data_train.T:
        c = Counter(feature)
        max_count = np.array([v for k, v in sorted(c.items())])[0]
        if max_count > 5:
            features_to_remove.append(i)
        i += 1
    data_train = data_train[:, np.array([i for i in range(data_train.shape[1]) if i not in features_to_remove])]
    data_test = data_test[:, np.array([i for i in range(data_test.shape[1]) if i not in features_to_remove])]

    #N = data_train.shape[0]
    # N_validate = int(N*0.1)
    # data_validate = data_train[-N_validate:]
    # data_train = data_train[0:-N_validate]
    # y_validate = y_train[-N_validate:]
    # y_train = y_train[0:-N_validate]
    return np.asarray(data_train.astype(np.float32)), np.asarray(data_test.astype(np.float32)), y_train, y_test
