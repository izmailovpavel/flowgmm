import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from oil.model_trainers.classifier import Classifier,Trainer
from oil.utils.losses import softmax_mse_loss, softmax_mse_loss_both
from oil.utils.utils import Eval, izip, icycle,imap
#from .schedules import sigmoidConsRamp
import flow_ssl
from flow_ssl.realnvp import RealNVPTabular
from flow_ssl.distributions import SSLGaussMixture
from scipy.spatial.distance import cdist
from train_semisup_flowgmm_tabular import RealNVPTabularWPrior,SemiFlow
from oil.tuning.study import Study, train_trial
import collections
import os
import utils
import copy
from train_semisup_text_baselines import makeTabularTrainer
from flow_ssl.data.nlp_datasets import AG_News,YAHOO
from flow_ssl.data import GAS, HEPMASS, MINIBOONE

trial = train_trial(makeTabularTrainer,strict=True)
text_flowgmm_cfg = {'num_epochs':30,'trainer':SemiFlow,'network':RealNVPTabularWPrior,'opt_config':{'lr':[1e-4]},
                'dataset':YAHOO,'unlab_loader_config':{'batch_size':lambda cfg: cfg['loader_config']['lab_BS']},#[MINIBOONE,HEPMASS,AG_News],
                'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/AG_News/flowgmm/'),'unlab_weight':[1.]},
                'loader_config': {'amnt_labeled':800+5000,'amnt_dev':5000,'lab_BS':800},'net_config':{'k':[512],'coupling_layers':[7],'nperlayer':1}}
uci_hepmass_flowgmm_cfg = {'num_epochs':20,'trainer':SemiFlow,'network':RealNVPTabularWPrior,'opt_config':{'lr':[3e-3,1e-3,1e-4]},
                'dataset':HEPMASS,#[MINIBOONE,HEPMASS,AG_News],
                'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/UCI/flowgmm/'),'unlab_weight':1.},
                'loader_config': {'amnt_labeled':20+5000,'amnt_dev':5000,'lab_BS':20},'net_config':{'k':256,'coupling_layers':10,'nperlayer':1}}
one_flowgmm_cfg = {'num_epochs':20,'trainer':SemiFlow,'network':RealNVPTabularWPrior,'opt_config':{'lr':1e-4},
                'dataset':MINIBOONE,#[MINIBOONE,HEPMASS,AG_News],
                'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/UCI/flowgmm/'),'unlab_weight':1.},
                'loader_config': {'amnt_labeled':20+5000,'amnt_dev':5000,'lab_BS':20},'net_config':{'k':256,'coupling_layers':5,'nperlayer':1}}
uci_mini_flowgmm_cfg = {'num_epochs':50,'trainer':SemiFlow,'network':RealNVPTabularWPrior,'opt_config':{'lr':3e-4},
                'dataset':MINIBOONE,#[MINIBOONE,HEPMASS,AG_News],
                'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/UCI/flowgmm/miniboone/'),'unlab_weight':1.},
                'loader_config': {'amnt_labeled':20+5000,'amnt_dev':5000,'lab_BS':20},'net_config':{'k':256,'coupling_layers':7,'nperlayer':1}}

#trial(one_flowgmm_cfg)
thestudy = Study(trial,text_flowgmm_cfg,study_name='text_hypers')
thestudy.run(3)
covars = thestudy.covariates()
covars['test_Acc'] = thestudy.outcomes['test_Acc'].values
covars['dev_Acc'] = thestudy.outcomes['dev_Acc'].values
print(covars.drop(['log_suffix','saved_at'],axis=1))
# print(thestudy.covariates())
# print(thestudy.outcomes)
