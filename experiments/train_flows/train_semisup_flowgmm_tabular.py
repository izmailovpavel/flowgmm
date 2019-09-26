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

def RealNVPTabularWPrior(num_classes,dim_in,coupling_layers,k,means_r=1.,cov_std=1.,nperlayer=1):
    #print(f'Instantiating means with dimension {dim_in}.')
    device = torch.device('cuda')
    inv_cov_std = torch.ones((num_classes,), device=device) / cov_std
    model = RealNVPTabular(num_coupling_layers=coupling_layers,in_dim=dim_in,hidden_dim=k,num_layers=1)#*np.sqrt(1000/dim_in)/3
    means = utils.get_means('random',r=means_r,num_means=num_classes, trainloader=None,shape=(dim_in),device=device)
    #means[1] = -means[0]
    model.prior = SSLGaussMixture(means, inv_cov_std,device=device)
    means_np = means.cpu().numpy()
    #print("Pairwise dists:", cdist(means_np, means_np))
    return model

class SemiFlow(Trainer):
    def __init__(self, *args, unlab_weight=1.,
                     **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers.update({'unlab_weight':unlab_weight})
        self.dataloaders['train'] = izip(icycle(self.dataloaders['lab']),self.dataloaders['_unlab'])

    def loss(self, minibatch):
        (x_lab, y_lab), x_unlab = minibatch
        # lab_loss = nn.CrossEntropyLoss()(self.model(x_lab),y_lab)
        #logits_labeled = self.model.prior.class_logits(self.model(x_lab))
        #lab_loss_nll = F.cross_entropy(logits_labeled, y_lab)
        lab_loss_nll = self.model.nll(x_lab,y_lab).mean()
        return lab_loss_nll + float(self.hypers['unlab_weight'])*self.model.nll(x_unlab).mean()/x_unlab.shape[1]

    def logStuff(self, step, minibatch=None):
        bpd_func = lambda mb: (self.model.nll(mb).mean().cpu().data.numpy()/mb.shape[-1] + np.log(256))/np.log(2)
        acc_func = lambda mb: self.model.prior.classify(self.model(mb[0])).type_as(mb[1]).eq(mb[1]).cpu().data.numpy().mean()
        metrics = {}
        with Eval(self.model):
            #metrics['Train_bpd'] = self.evalAverageMetrics(self.dataloaders['unlab'],bpd_func)
            metrics['Dev_bpd'] = self.evalAverageMetrics(imap(lambda z: z[0],self.dataloaders['dev']),bpd_func)
            metrics['Train_Acc'] = self.evalAverageMetrics(self.dataloaders['Train'],acc_func)
            metrics['dev_Acc'] = self.evalAverageMetrics(self.dataloaders['dev'],acc_func)
            metrics['test_Acc'] = self.evalAverageMetrics(self.dataloaders['test'],acc_func)
            if minibatch:
                metrics['Unlab_loss(mb)']=self.model.nll(minibatch[1]).mean().cpu().data.numpy()
        self.logger.add_scalars('metrics',metrics,step)
        super().logStuff(step, minibatch)


from oil.tuning.study import Study, train_trial
import collections
import os
import utils
import copy
from train_semisup_text_baselines import makeTabularTrainer
from flow_ssl.data.nlp_datasets import AG_News
from flow_ssl.data import GAS, HEPMASS, MINIBOONE

trial = train_trial(makeTabularTrainer,strict=True)

text_flowgmm_cfg = {'num_epochs':200,'trainer':SemiFlow,'network':RealNVPTabularWPrior,'opt_config':{'lr':[1e-3,3e-3,1e-4]},
                'dataset':AG_News,#[MINIBOONE,HEPMASS,AG_News],
                'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/AG_News/flowgmm/'),'unlab_weight':[100.,10.,1.]},
                'loader_config': {'amnt_labeled':200+5000,'amnt_dev':5000,'lab_BS':200},'net_config':{'k':[512,256],'coupling_layers':[5,7,10],'nperlayer':[1,2]}}
uci_hepmass_flowgmm_cfg = {'num_epochs':50,'trainer':SemiFlow,'network':RealNVPTabularWPrior,'opt_config':{'lr':1e-3},
                'dataset':HEPMASS,#[MINIBOONE,HEPMASS,AG_News],
                'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/UCI/flowgmm/'),'unlab_weight':1.},
                'loader_config': {'amnt_labeled':20+5000,'amnt_dev':5000,'lab_BS':20},'net_config':{'k':256,'coupling_layers':10,'nperlayer':1}}
one_flowgmm_cfg = {'num_epochs':20,'trainer':SemiFlow,'network':RealNVPTabularWPrior,'opt_config':{'lr':3e-3},
                'dataset':MINIBOONE,#[MINIBOONE,HEPMASS,AG_News],
                'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/UCI/flowgmm/'),'unlab_weight':1.},
                'loader_config': {'amnt_labeled':20+5000,'amnt_dev':5000,'lab_BS':20},'net_config':{'k':256,'coupling_layers':5,'nperlayer':1}}
uci_mini_flowgmm_cfg = {'num_epochs':50,'trainer':SemiFlow,'network':RealNVPTabularWPrior,'opt_config':{'lr':3e-4},
                'dataset':MINIBOONE,#[MINIBOONE,HEPMASS,AG_News],
                'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/UCI/flowgmm/miniboone/'),'unlab_weight':1.},
                'loader_config': {'amnt_labeled':20+5000,'amnt_dev':5000,'lab_BS':20},'net_config':{'k':256,'coupling_layers':7,'nperlayer':1}}
if __name__=="__main__":
    #trial(one_flowgmm_cfg)
    thestudy = Study(trial,uci_hepmass_flowgmm_cfg,study_name='uci_flowgmm_hypers222_m')
    thestudy.run(3,ordered=False)
    covars = thestudy.covariates()
    covars['test_Acc'] = thestudy.outcomes['test_Acc'].values
    covars['dev_Acc'] = thestudy.outcomes['dev_Acc'].values
    print(covars.drop(['log_suffix','saved_at'],axis=1))
    # print(thestudy.covariates())
    # print(thestudy.outcomes)
