import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from oil.model_trainers.classifier import Classifier,Trainer
from oil.utils.losses import softmax_mse_loss, softmax_mse_loss_both
from oil.utils.utils import Eval, izip, icycle,imap, export
#from .schedules import sigmoidConsRamp
import flow_ssl
import utils
from flow_ssl import FlowLoss
from flow_ssl.realnvp import RealNVPTabular
from flow_ssl.distributions import SSLGaussMixture
from scipy.spatial.distance import cdist

@export
def RealNVPTabularWPrior(num_classes,dim_in,coupling_layers,k,means_r=.8,cov_std=1.,nperlayer=1,acc=0.9):
    #print(f'Instantiating means with dimension {dim_in}.')
    device = torch.device('cuda')
    inv_cov_std = torch.ones((num_classes,), device=device) / cov_std
    model = RealNVPTabular(num_coupling_layers=coupling_layers,in_dim=dim_in,hidden_dim=k,num_layers=1,dropout=True)#*np.sqrt(1000/dim_in)/3
    #dist_scaling = np.sqrt(-8*np.log(1-acc))#np.sqrt(4*np.log(20)/dim_in)#np.sqrt(1000/dim_in)
    if num_classes ==2:
        means = utils.get_means('random',r=means_r,num_means=num_classes, trainloader=None,shape=(dim_in),device=device)
        #means = torch.zeros(2,dim_in,device=device)
        #means[0,1] = 3.75
        dist = 2*(means[0]**2).sum().sqrt()
        means[0] *= 7.5/dist
        means[1] = -means[0]
        # means[0] /= means[0].norm()
        # means[0] *= dist_scaling/2
        # means[1] = - means[0]
        model.prior = SSLGaussMixture(means, inv_cov_std,device=device)
        means_np = means.cpu().numpy()
    else:
        means = utils.get_means('random',r=means_r*.7,num_means=num_classes, trainloader=None,shape=(dim_in),device=device)
        model.prior = SSLGaussMixture(means, inv_cov_std,device=device)
        means_np = means.cpu().numpy()
    print("Pairwise dists:", cdist(means_np, means_np))
    return model

def ResidualTabularWPrior(num_classes,dim_in,coupling_layers,k,means_r=1.,cov_std=1.,nperlayer=1,acc=0.9):
    #print(f'Instantiating means with dimension {dim_in}.')
    device = torch.device('cuda')
    inv_cov_std = torch.ones((num_classes,), device=device) / cov_std
    model = TabularResidualFlow(in_dim=dim_in,hidden_dim=k,num_per_block=coupling_layers)#*np.sqrt(1000/dim_in)/3
    dist_scaling = np.sqrt(-8*np.log(1-acc))
    means = utils.get_means('random',r=means_r*dist_scaling,num_means=num_classes, trainloader=None,shape=(dim_in),device=device)
    means[0] /= means[0].norm()
    means[0] *= dist_scaling/2
    means[1] = - means[0]
    model.prior = SSLGaussMixture(means, inv_cov_std,device=device)
    means_np = means.cpu().numpy()
    #print("Pairwise dists:", cdist(means_np, means_np))
    return model

@export
class SemiFlow(Trainer):
    def __init__(self, *args, unlab_weight=1.,cons_weight=3.,
                     **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers.update({'unlab_weight':unlab_weight,'cons_weight':cons_weight})
        self.dataloaders['train'] = izip(icycle(self.dataloaders['train']),self.dataloaders['_unlab'])

    def loss(self, minibatch):
        (x_lab, y_lab), x_unlab = minibatch
        a = float(self.hypers['unlab_weight'])
        b = float(self.hypers['cons_weight'])
        flow_loss = self.model.nll(x_lab,y_lab).mean() + a*self.model.nll(x_unlab).mean()
        # with torch.no_grad():
        #     unlab_label = self.model.prior.classify(self.model(x_unlab)).detach()
        # cons_loss = self.model.nll(x_unlab,unlab_label).mean()
        return flow_loss#+b*cons_loss
    def step(self, minibatch):
        self.optimizer.zero_grad()
        loss = self.loss(minibatch)
        loss.backward()
        utils.clip_grad_norm(self.optimizer, 100)
        self.optimizer.step()
        return loss
        
    def logStuff(self, step, minibatch=None):
        bpd_func = lambda mb: (self.model.nll(mb).mean().cpu().data.numpy()/mb.shape[-1] + np.log(256))/np.log(2)
        acc_func = lambda mb: self.model.prior.classify(self.model(mb[0])).type_as(mb[1]).eq(mb[1]).cpu().data.numpy().mean()
        metrics = {}
        with Eval(self.model), torch.no_grad():
            #metrics['Train_bpd'] = self.evalAverageMetrics(self.dataloaders['unlab'],bpd_func)
            metrics['val_bpd'] = self.evalAverageMetrics(imap(lambda z: z[0],self.dataloaders['val']),bpd_func)
            metrics['Train_Acc'] = self.evalAverageMetrics(self.dataloaders['Train'],acc_func)
            metrics['val_Acc'] = self.evalAverageMetrics(self.dataloaders['val'],acc_func)
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
#from train_semisup_text_baselines import makeTabularTrainer
#from flowgmm_tabular_new import tabularTrial
from flow_ssl.data.nlp_datasets import AG_News
from flow_ssl.data import GAS, HEPMASS, MINIBOONE

# if __name__=="__main__":
#     trial(uci_hepmass_flowgmm_cfg)
    # thestudy = Study(trial,uci_hepmass_flowgmm_cfg,study_name='uci_flowgmm_hypers222_m__m_m')
    # thestudy.run(1,ordered=False)
    # covars = thestudy.covariates()
    # covars['test_Acc'] = thestudy.outcomes['test_Acc'].values
    # covars['dev_Acc'] = thestudy.outcomes['dev_Acc'].values
    #print(covars.drop(['log_suffix','saved_at'],axis=1))
    # print(thestudy.covariates())
    # print(thestudy.outcomes)

