import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import itertools
import os
import torch.nn.functional as F
from oil.model_trainers.trainer import Trainer
from oil.model_trainers.classifier import Classifier
import scipy.misc
from itertools import islice
import numpy as np
from oil.utils.utils import Eval

class Flow(Trainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.fixed_input = (self.G.sample_z(32),)

    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        x,y = minibatch
        return self.model.nll(x).mean()/np.prod(x.shape[1:])

    def logStuff(self, step, minibatch=None):
        """ Handles Logging and any additional needs for subclasses,
            should have no impact on the training """

        # metrics = {}
        # try: metrics['FID'],metrics['IS'] = FID_and_IS(self.as_dataloader(),self.dataloaders['dev'])
        # except KeyError: pass
        # self.logger.add_scalars('metrics', metrics, step)
        # if hasattr(self.model,'sample') and minibatch is not None:
        #     with Eval(self.model):
        #         self.model.nll(minibatch[0]) # Forward through network to populate shape info
        #         with torch.no_grad():
        #             fake_images = self.model.sample(32).cpu().data
        #     img_grid = vutils.make_grid(fake_images, normalize=False,range=(0,1))
        #     self.logger.add_image('samples', img_grid, step)
        super().logStuff(step,minibatch=None)

    def metrics(self,loader):
        nll_func = lambda mb: self.loss(mb).cpu().data.numpy()
        with Eval(self.model):
            nll = self.evalAverageMetrics(loader,nll_func)
        return {'bpd':(nll + np.log(256))/np.log(2)}

class iClassifier(Classifier):
    
    def __init__(self, *args, ld_weight=1000.,kp=1/2, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers['ld_weight'] = ld_weight
        self.hypers['kp'] = kp
        self.rel_err=1
        #self.fixed_input = (self.G.sample_z(32),)
    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        x,y = minibatch
        CE = super().loss(minibatch)
        barrier_function = lambda x: 1/x + x**2
        p = self.rel_err**self.hypers['kp']
        return  CE + self.hypers['ld_weight']*p*self.model[1].body.reduce_func_singular_values(barrier_function).mean()/np.prod(x.shape[1:])
        #return  CE - self.hypers['ld_weight']*self.model[1].body.logdet().mean()/np.prod(x.shape[1:])

    def logStuff(self, step, minibatch=None):
        """ Handles Logging and any additional needs for subclasses,
            should have no impact on the training """
        with Eval(self.model), torch.no_grad():
            if minibatch is not None:
                z = self.model[1].body(minibatch[0])
                reconstructed = self.model[1].body.inverse(z)
                rel_err = (torch.mean((minibatch[0]-reconstructed)**2).sqrt()\
                    /torch.mean(minibatch[0]**2).sqrt()).cpu().data.numpy()
                if rel_err<0.03:
                    self.hypers['ld_weight']*=.8
                elif rel_err>0.06:
                    self.hypers['ld_weight']*=1.25
                self.rel_err = rel_err
                p = self.rel_err**self.hypers['kp']
                self.logger.add_scalars('info', {'recons_err':rel_err,'ld_weight':self.hypers['ld_weight'],'p':p}, step)
        # if hasattr(self.model,'sample') and minibatch is not None:
        #     with Eval(self.model):
        #         self.model.nll(minibatch[0]) # Forward through network to populate shape info
        #         with torch.no_grad():
        #             fake_images = self.model.sample(32).cpu().data
        #     img_grid = vutils.make_grid(fake_images, normalize=False,range=(0,1))
        #     self.logger.add_image('samples', img_grid, step)
        super().logStuff(step,minibatch)

from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, recursively_update,islice
from oil.tuning.study import train_trial
from oil.datasetup.dataloaders import getLabLoader
from oil.datasetup.datasets import CIFAR10
from oil.architectures.img_classifiers import layer13s
#from invertible.iresnet import iResnet,iResnetLarge
from flow_ssl.icnn.icnn import iCNN
from flow_ssl.iresnet import ResidualFlow
from oil.utils.parallel import MyDataParallel, MyDistributedDataParallel,multigpu_parallelize
import torchvision.transforms as transforms
import collections


def simpleiClassifierTrial(strict=False):
    def makeTrainer(config):
        cfg = {
            'dataset': CIFAR10,'network':layer13s,'net_config': {},
            'loader_config': {'amnt_dev':5000,'lab_BS':20, 'pin_memory':True,'num_workers':2},
            'opt_config':{'lr':.3e-4},#, 'momentum':.9, 'weight_decay':1e-4,'nesterov':True},
            'num_epochs':100,'trainer_config':{},
            }
        recursively_update(cfg,config)
        trainset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']),flow=True)
        device = torch.device('cuda')
        fullCNN = torch.nn.Sequential(
            trainset.default_aug_layers(),
            cfg['network'](num_classes=trainset.num_classes,**cfg['net_config']).to(device)
        )
        dataloaders = {}
        dataloaders['train'], dataloaders['dev'] = getLabLoader(trainset,**cfg['loader_config'])
        dataloaders['Train'] = islice(dataloaders['train'],10000//cfg['loader_config']['lab_BS'])
        if len(dataloaders['dev'])==0:
            testset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']),train=False)
            dataloaders['test'] = DataLoader(testset,batch_size=cfg['loader_config']['lab_BS'],shuffle=False)
        dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}
        opt_constr = lambda params: torch.optim.Adam(params, **cfg['opt_config'])#torch.optim.SGD(params, **cfg['opt_config'])
        lr_sched = cosLr(cfg['num_epochs'])
        return iClassifier(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])
    return train_trial(makeTrainer,strict)


def simpleFlowTrial(strict=False):
    def makeTrainer(config):
        cfg = {
            'dataset': CIFAR10,'network':iCNN,'net_config':{},
            'loader_config': {'amnt_dev':5000,'lab_BS':32, 'pin_memory':True,'num_workers':3},
            'opt_config':{'lr':.0003,},# 'momentum':.9, 'weight_decay':1e-4,'nesterov':True},
            'num_epochs':100,'trainer_config':{},'parallel':False,
            }
        recursively_update(cfg,config)
        train_transforms = transforms.Compose([transforms.ToTensor(),transforms.RandomHorizontalFlip()])
        trainset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']),flow=True,)
        device = torch.device('cuda')
        fullCNN = cfg['network'](num_classes=trainset.num_classes,**cfg['net_config']).to(device)
        if cfg['parallel']: fullCNN = multigpu_parallelize(fullCNN,cfg)
        dataloaders = {}
        dataloaders['train'], dataloaders['dev'] = getLabLoader(trainset,**cfg['loader_config'])
        dataloaders['Train'] = islice(dataloaders['train'],10000//cfg['loader_config']['lab_BS'])
        if len(dataloaders['dev'])==0:
            testset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']),train=False,flow=True)
            dataloaders['test'] = DataLoader(testset,batch_size=cfg['loader_config']['lab_BS'],shuffle=False)
        dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}#LoaderTo(v,device)
        opt_constr = lambda params: torch.optim.Adam(params, **cfg['opt_config'])
        lr_sched = cosLr(cfg['num_epochs'])
        return Flow(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])
    return train_trial(makeTrainer,strict)

if __name__=='__main__':
    Trial = simpleFlowTrial(strict=True)
    Trial({'num_epochs':100,'network':ResidualFlow,'opt_config':{'lr':.0001},
    'net_config':{'k':512,'num_per_block':12},'trainer_config':{'log_args':{'timeFrac':1/2}}})