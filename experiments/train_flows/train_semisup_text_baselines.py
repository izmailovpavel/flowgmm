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

@export
class SmallNN(nn.Module,metaclass=Named):
    """
    Very small CNN
    """
    def __init__(self, dim_in=768, num_classes=4,k=512):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(dim_in,k),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(k,k),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(k,num_classes),
        )
    def forward(self,x):
        return self.net(x)


# def makeTrainer(config):
#     cfg = {'network':SmallNN,'net_config': {},
#         'loader_config': {'amnt_labeled':200+5000,'amnt_dev':5000,'lab_BS':200, 'pin_memory':True,'num_workers':2},
#         'opt_config':{'lr':1e-3},#, 'momentum':.9, 'weight_decay':1e-4,'nesterov':True},
#         'num_epochs':100,
#         'trainer':PiModel,
#         'unlab_loader_config':{'batch_size':2000,'num_workers':4,'pin_memory':True},
#         'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/text_vat/'),'log_args':{'minPeriod':.1, 'timeFrac':3/10}},
#         }
#     recursively_update(cfg,config)
#     trainset = AG_News(os.path.expanduser('~/datasets/AGNEWS/'),train=True)
#     testset = AG_News(os.path.expanduser('~/datasets/AGNEWS/'),train=False)
#     device = torch.device('cpu')
#     model = cfg['network'](num_classes=trainset.num_classes,dim_in=trainset.dim,**cfg['net_config']).to(device)
#     dataloaders = {}
#     dataloaders['lab'], dataloaders['dev'] = getLabLoader(trainset,**cfg['loader_config'])
#     dataloaders['train'] = dataloaders['Train'] = dataloaders['lab']
    
#     full_data_loader = DataLoader(trainset,shuffle=True,**cfg['unlab_loader_config'])
#     dataloaders['_unlab'] = imap(lambda z: z[0], full_data_loader)
#     dataloaders['test'] = DataLoader(testset,batch_size=cfg['loader_config']['lab_BS'],shuffle=False)
#     dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}
#     opt_constr = lambda params: torch.optim.Adam(params, **cfg['opt_config'])
#     lr_sched = cosLr(cfg['num_epochs'])
#     return cfg['trainer'](model,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])


def makeTabularTrainer(**config):
    cfg = {'dataset':HEPMASS,'network':SmallNN,'net_config': {},
        'loader_config': {'amnt_labeled':20+5000,'amnt_dev':5000,'lab_BS':20},
        'opt_config': {'lr':1e-4},#{'lr':.1, 'momentum':.9, 'weight_decay':1e-4, 'nesterov':True},
        'num_epochs':200,
        'unlab_loader_config':{'batch_size':2000,'num_workers':4,'pin_memory':True},
        'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/UCI/'),'log_args':{'minPeriod':.1, 'timeFrac':3/10}},
        }
    recursively_update(cfg,config)
    
    trainset = cfg['dataset'](train=True)
    testset = cfg['dataset'](train=False)
    print(f"Trainset: {len(trainset)}, Testset: {len(testset)}")
    device = torch.device('cuda')
    model = cfg['network'](num_classes=trainset.num_classes,dim_in=trainset.dim,**cfg['net_config']).to(device)
    dataloaders = {}
    dataloaders['lab'], dataloaders['dev'] = getLabLoader(trainset,**cfg['loader_config'])
    dataloaders['train'] = dataloaders['Train'] = dataloaders['lab']
    
    full_data_loader = DataLoader(trainset,shuffle=True,**cfg['unlab_loader_config'])
    dataloaders['_unlab'] = imap(lambda z: z[0], full_data_loader)
    dataloaders['test'] = DataLoader(testset,batch_size=cfg['loader_config']['lab_BS'],shuffle=False)
    dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}
    opt_constr = lambda params: torch.optim.Adam(params, **cfg['opt_config'])
    lr_sched = lambda e: 1.#cosLr(cfg['num_epochs'])
    return cfg['trainer'](model,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])


PI_trial = train_trial(makeTabularTrainer,strict=True)

uci_pi_spec = {'network':SmallNN,'net_config': {},'dataset':[MINIBOONE,HEPMASS],
        #'loader_config': {'lab_BS':200},
        'opt_config': {'lr':[1e-3,3e-3,1e-4]},
        'loader_config': {'amnt_labeled':20+5000,'amnt_dev':5000,'lab_BS':20},
        'num_epochs':50,#100,#100,#5,#800,
        #'unlab_loader_config':{'batch_size':2000},
        'net_config':{'k':[256,512]},'trainer':PiModel,
        'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/UCI/t3layer_pi_uci3/'),
        'cons_weight':[20,30,50]}#[1,.1,.3,3],}#'advEps':[10,3,1,.3]}
        }
uci_pi_spec2 = {'network':SmallNN,'net_config': {},'dataset':[MINIBOONE],
        #'loader_config': {'lab_BS':200},
        'opt_config': {'lr':lambda cfg: 3e-3 if cfg['dataset']==HEPMASS else 3e-5},
        'loader_config': {'amnt_labeled':20+5000,'amnt_dev':5000,'lab_BS':20},
        'num_epochs':50,#100,#100,#5,#800,
        #'unlab_loader_config':{'batch_size':2000},
        'net_config':{'k':256},'trainer':PiModel,
        'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/UCI/t3layer_pi_uci3r/'),
        'cons_weight':30}#[1,.1,.3,3],}#'advEps':[10,3,1,.3]}
        }
uci_baseline_spec = {'network':SmallNN,'net_config': {},'dataset':[MINIBOONE],
        #'loader_config': {'lab_BS':200},
        'opt_config': {'lr':3e-4},
        'loader_config': {'amnt_labeled':20+5000,'amnt_dev':5000,'lab_BS':20},
        'num_epochs':100,#100,#100,#5,#800,
        #'unlab_loader_config':{'batch_size':2000},
        'net_config':{'k':256},'trainer':Classifier,
        'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/UCI/t3layer_baseline/'),
        'log_args':{'minPeriod':.01, 'timeFrac':3/10}}#[1,.1,.3,3],}#'advEps':[10,3,1,.3]}
        }
if __name__=='__main__':
    # thestudy = Study(PI_trial,uci_pi_spec2,study_name='uci_baseline2234_')
    # thestudy.run(num_trials=3,ordered=False)
    # #print(thestudy.covariates())
    # covars = thestudy.covariates()
    # covars['test_Acc'] = thestudy.outcomes['test_Acc'].values
    # covars['dev_Acc'] = thestudy.outcomes['dev_Acc'].values
    # print(covars.drop(['log_suffix','saved_at'],axis=1))

    # PI model baselines for AG-NEWS w/ best hyperparameters
    text_pi_cfg = {'dataset':AG_News,'num_epochs':50,'trainer':PiModel,'trainer_config':{'cons_weight':30},'opt_config':{'lr':1e-3},
                    'loader_config': {'amnt_labeled':200+5000,'lab_BS':200}}
    text_classifier_cfg = {'dataset':AG_News,'num_epochs':500,'trainer':Classifier,'opt_config':{'lr':1e-3},
                    'loader_config': {'amnt_labeled':200+5000,'lab_BS':200}}
    y_text_pi_cfg = {'dataset':YAHOO,'num_epochs':50,'trainer':PiModel,'trainer_config':{'cons_weight':[30]},'opt_config':{'lr':[1e-4]},
                    'loader_config': {'amnt_labeled':800+5000,'lab_BS':800}}
    y_text_classifier_cfg = {'dataset':YAHOO,'num_epochs':500,'trainer':Classifier,'opt_config':{'lr':1e-3},
                    'loader_config': {'amnt_labeled':800+5000,'lab_BS':800}}
    # Searched from
    # text_pi_cfg = {'num_epochs':50,'trainer':PiModel,'trainer_config':{'cons_weight':[10,30]},'opt_config':{'lr':[1e-3,3e-4,3e-3]},
    #                   'loader_config': {'amnt_labeled':200+5000,'lab_BS':200}}

    textstudy = Study(PI_trial,y_text_pi_cfg,study_name='Agnews')
    textstudy.run(3)
    print(textstudy.covariates())
    print(textstudy.outcomes)