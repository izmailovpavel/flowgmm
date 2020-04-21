import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
#import oil.augLayers as augLayers
from oil.model_trainers.classifier import Classifier
from oil.model_trainers.piModel import PiModel
from oil.model_trainers.vat import Vat
from oil.datasetup.datasets import CIFAR10
from oil.datasetup.dataloaders import getLabLoader
from oil.architectures.img_classifiers.networkparts import layer13
from oil.utils.utils import LoaderTo, cosLr, recursively_update,islice, imap
from oil.tuning.study import Study, train_trial
from flow_ssl.data.nlp_datasets import AG_News,YAHOO
from flow_ssl.data import GAS, HEPMASS, MINIBOONE
from torchvision import transforms
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from oil.utils.utils import Expression,export,Named
from collections import defaultdict
from oil.model_trainers.piModel import PiModel
from oil.model_trainers.classifier import Classifier
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD,Adam,AdamW
from oil.utils.utils import LoaderTo, cosLr, islice, dmap, FixedNumpySeed
from oil.tuning.study import train_trial
from oil.datasetup.datasets import CIFAR10, split_dataset
from oil.tuning.args import argupdated_config
from functools import partial
from train_semisup_text_baselines import SmallNN
from oil.tuning.args import argupdated_config
import copy
#import flow_ssl.data.nlp_datasets as nlp_datasets
import flow_ssl.data as tabular_datasets
import train_semisup_flowgmm_tabular as flows
import train_semisup_text_baselines as archs
import oil.model_trainers as trainers
def makeTrainer(*,dataset=HEPMASS,network=SmallNN,num_epochs=15,
                bs=5000,lr=1e-3,optim=AdamW,device='cuda',trainer=Classifier,
                split={'train':20,'val':5000},net_config={},opt_config={'weight_decay':1e-5},
                trainer_config={'log_dir':os.path.expanduser('~/tb-experiments/UCI/'),'log_args':{'minPeriod':.1, 'timeFrac':3/10}},
                save=False):

    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(0):
        datasets = split_dataset(dataset(),splits=split)
        datasets['_unlab'] = dmap(lambda mb: mb[0],dataset())
        datasets['test'] = dataset(train=False)
        #print(datasets['test'][0])
    device = torch.device(device)
    model = network(num_classes=datasets['train'].num_classes,dim_in=datasets['train'].dim,**net_config).to(device)

    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(datasets[k])),shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)#lambda e:1#
    return trainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)

tabularTrial = train_trial(makeTrainer)

if __name__=='__main__':
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    cfg = argupdated_config(defaults,namespace=(tabular_datasets,flows,archs,trainers))
    cfg.pop('local_rank')
    trainer = makeTrainer(**cfg)
    #tabularTrial()
    trainer.train(cfg['num_epochs'])
