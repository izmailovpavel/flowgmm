import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os.path
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pickle

class GAS(Dataset):
    num_classes=2
    class_weights=None
    ignored_index=-100
    def __init__(self,root=os.path.expanduser('~/datasets/UCI/gas/'),train=True,remake=False):
        super().__init__()
        if os.path.exists(root+'dataset.pickle') and not remake:
            with open(root+'dataset.pickle','rb') as f:
                self.__dict__ = pickle.load(f).__dict__
        else:
            X,Y = load_data_and_clean(root)
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

    # def show_histograms(self, split):

    #     data_split = getattr(self, split, None)
    #     if data_split is None:
    #         raise ValueError('Invalid data split')

    #     util.plot_hist_marginals(data_split.x)
    #     plt.show()


def data_iter(file):
    with open(file,'r') as f:
        for i,line in enumerate(f.readlines()):
            #print(line)
            if i==0: continue
            try: yield [float(val) for val in line.split(' ') if val not in ('','\n')][3:19]
            except ValueError:
                print([val for val in line.split(' ') if val!=''][3:19])
                assert False

#np.save(os.path.expanduser('~/datasets/UCI/gas/ethylene_CO'),np.vstack(data_iter(os.path.expanduser('~/datasets/UCI/ethylene_CO.txt'))))
#np.save(os.path.expanduser('~/datasets/UCI/gas/ethylene_methane'),np.vstack(data_iter(os.path.expanduser('~/datasets/UCI/ethylene_methane.txt'))))
def load_data(file):
    data = pd.read_pickle(file)
    # data = pd.read_pickle(file).sample(frac=0.25)
    # data.to_pickle(file)
    data.drop("Meth", axis=1, inplace=True)
    data.drop("Eth", axis=1, inplace=True)
    data.drop("Time", axis=1, inplace=True)
    return data


def get_correlation_numbers(data):
    C = data.corr()
    A = C > 0.98
    B = A.values.sum(axis=1)
    return B


def load_data_and_clean(root):

    co_data = pd.DataFrame(np.load(root+'ethylene_CO.npy')).sample(frac=0.1)
    methane_data = pd.DataFrame(np.load(root+'ethylene_methane.npy')).sample(frac=0.1)
    nco = len(co_data)
    nmeth = len(methane_data)
    data = pd.concat((co_data,methane_data))
    B = get_correlation_numbers(data)
    labels = np.concatenate((np.zeros(nco),np.ones(nmeth)))
    while np.any(B > 1):
        col_to_remove = np.where(B > 1)[0][0]
        col_name = data.columns[col_to_remove]
        data.drop(col_name, axis=1, inplace=True)
        B = get_correlation_numbers(data)
    # print(data.corr())
    data = (data-data.mean(0))/data.std(0)

    return data.values, labels