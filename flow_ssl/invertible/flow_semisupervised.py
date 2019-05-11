import torch
import numpy as np
import torch.nn as nn
from oil.model_trainers.classifier import Classifier,Trainer
from oil.utils.losses import softmax_mse_loss, softmax_mse_loss_both
from oil.utils.utils import Eval, izip, icycle,imap
from flow import Flow
#from .schedules import sigmoidConsRamp

class SemiFlow(Trainer):
    def __init__(self, *args, unlab_weight=1.,
                     **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers.update({'unlab_weight':unlab_weight})
        self.dataloaders['train'] = izip(icycle(self.dataloaders['lab']),self.dataloaders['unlab'])

    def unlabLoss(self, x_unlab):
        z = self.model.get_all_z_squashed(x_unlab)
        l2 = .5*(z**2).sum()/len(z) + .5*np.log(2*np.pi)
        lndet = self.model.logdet().sum()/len(z)
        nll = l2-lndet#+l2#l2 + lndets
        return nll

    def loss(self, minibatch):
        (x_lab, y_lab), x_unlab = minibatch
        lab_loss = nn.CrossEntropyLoss()(self.model(x_lab),y_lab)
        unlab_loss = self.unlabLoss(x_unlab)*float(self.hypers['unlab_weight'])
        return lab_loss + unlab_loss

    def logStuff(self, step, minibatch=None):
        bpd_func = lambda mb: (self.unlabLoss(mb).cpu().data.numpy() + np.log(256))/np.log(2)
        acc_func = lambda mb: self.model(mb[0]).max(1)[1].type_as(mb[1]).eq(mb[1]).cpu().data.numpy().mean()
        metrics = {}
        with Eval(self.model):
            #metrics['Train_bpd'] = self.evalAverageMetrics(self.dataloaders['unlab'],bpd_func)
            metrics['Dev_bpd'] = self.evalAverageMetrics(imap(lambda z: z[0],self.dataloaders['dev']),bpd_func)
            metrics['Train_acc'] = self.evalAverageMetrics(self.dataloaders['Train'],acc_func)
            metrics['Dev_acc'] = self.evalAverageMetrics(self.dataloaders['dev'],acc_func)
            if minibatch:
                metrics['Unlab_loss(mb)']=self.unlabLoss(minibatch[1]).cpu().data.numpy()
        self.logger.add_scalars('metrics',metrics,step)
        super().logStuff(step, minibatch)


from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, recursively_update,islice
from oil.tuning.study import train_trial
from oil.datasetup.dataloaders import getLabLoader
from oil.datasetup.datasets import CIFAR10
from oil.architectures.img_classifiers import layer13s
from iresnet import iResnet,iResnetLarge
from iEluNetwork import iEluNet,iEluNetMultiScale,iEluNetMultiScaleLarger
from oil.tuning.study import Study, train_trial
import collections
import os
import copy

def semiFlowTrial(strict=False):
    def makeTrainer(config):
        cfg = {
            'dataset': CIFAR10,'network':iEluNetMultiScaleLarger,
            'loader_config': {'amnt_labeled':4000+5000,'amnt_dev':5000,'lab_BS':50, 'pin_memory':True,'num_workers':2},
            'opt_config':{'lr':.001,},# 'momentum':.9, 'weight_decay':1e-4,'nesterov':True},
            'num_epochs':50,'trainer_config':{},
            'unlab_loader_config':{'batch_size':50,'num_workers':2},
            }
        recursively_update(cfg,config)
        trainset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']))
        device = torch.device('cuda')
        fullCNN = cfg['network'](num_classes=trainset.num_classes,**cfg['net_config']).to(device)
        
        dataloaders = {}
        dataloaders['lab'], dataloaders['dev'] = getLabLoader(trainset,**cfg['loader_config'])
        dataloaders['Train'] = islice(dataloaders['lab'],10000//cfg['loader_config']['lab_BS'])
        full_cifar_loader = DataLoader(trainset,shuffle=True,**cfg['unlab_loader_config'])
        dataloaders['unlab'] = imap(lambda z: z[0], full_cifar_loader)
        if len(dataloaders['dev'])==0:
            testset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']),train=False)
            dataloaders['test'] = DataLoader(testset,batch_size=cfg['loader_config']['lab_BS'],shuffle=False)
        dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}
        opt_constr = lambda params: torch.optim.Adam(params, **cfg['opt_config'])
        lr_sched = cosLr(cfg['num_epochs'])
        return SemiFlow(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])
    return train_trial(makeTrainer,strict)