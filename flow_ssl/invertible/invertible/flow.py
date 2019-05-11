import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import itertools
import os
import torch.nn.functional as F
from oil.model_trainers.trainer import Trainer
import scipy.misc
from itertools import islice
import numpy as np
from oil.utils.utils import Eval

class Flow(Trainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.fixed_input = (self.G.sample_z(32),)

    def loss(self, minibatch, model = None):
        """ Standard cross-entropy loss """
        x,y = minibatch
        if model is None: model = self.model
        with torch.autograd.enable_grad():
            #x.requires_grad = True
            z = model.get_all_z_squashed(x)
            
            # 32 x (32x32x32)
            l2 = .5*(z**2).sum()/len(z) + .5*np.log(2*np.pi)
            #l2.backward(retain_graph=True)
            lndet = model.logdet().sum()/len(z)
            
        nll = l2-lndet#+l2#l2 + lndets
        return nll#nll#nn.MSELoss()(z,torch.zeros_like(z))

    def logStuff(self, step, minibatch=None):
        """ Handles Logging and any additional needs for subclasses,
            should have no impact on the training """

        # metrics = {}
        # try: metrics['FID'],metrics['IS'] = FID_and_IS(self.as_dataloader(),self.dataloaders['dev'])
        # except KeyError: pass
        # self.logger.add_scalars('metrics', metrics, step)
        if hasattr(self.model,'sample') and minibatch is not None:
            with Eval(self.model), torch.no_grad():
                self.model(minibatch[0])
                fake_images = self.model.sample(32).cpu().data
                # for i,p in enumerate(self.model.parameters()):
                #     if (i>=5) and (i<8): 
                #         print(f"Some weight: {p.reshape(-1)[-1]}")
            img_grid = vutils.make_grid(fake_images, normalize=False,range=(0,1))
            self.logger.add_image('samples', img_grid, step)
        super().logStuff(step,minibatch)

    def metrics(self,loader):
        nll_func = lambda mb: self.loss(mb).cpu().data.numpy()
        with Eval(self.model):
            nll = self.evalAverageMetrics(loader,nll_func)
        return {'bpd':(nll + np.log(256))/np.log(2)}
    

from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, recursively_update,islice
from oil.tuning.study import train_trial
from oil.datasetup.dataloaders import getLabLoader
from oil.datasetup.datasets import CIFAR10
from oil.architectures.img_classifiers import layer13s
from invertible.iresnet import iResnet,iResnetLarge
import collections

def simpleFlowTrial(strict=False):
    def makeTrainer(config):
        cfg = {
            'dataset': CIFAR10,'network':iResnet,
            'loader_config': {'amnt_dev':5000,'lab_BS':64, 'pin_memory':True,'num_workers':2},
            'opt_config':{'lr':.0003,},# 'momentum':.9, 'weight_decay':1e-4,'nesterov':True},
            'num_epochs':100,'trainer_config':{},
            }
        recursively_update(cfg,config)
        trainset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']),flow=True)
        device = torch.device('cuda')
        fullCNN = cfg['network'](num_classes=trainset.num_classes,**cfg['net_config']).to(device)
        
        dataloaders = {}
        dataloaders['train'], dataloaders['dev'] = getLabLoader(trainset,**cfg['loader_config'])
        dataloaders['Train'] = islice(dataloaders['train'],10000//cfg['loader_config']['lab_BS'])
        if len(dataloaders['dev'])==0:
            testset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']),train=False,flow=True)
            dataloaders['test'] = DataLoader(testset,batch_size=cfg['loader_config']['lab_BS'],shuffle=False)
        dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}
        opt_constr = lambda params: torch.optim.Adam(params, **cfg['opt_config'])
        lr_sched = cosLr(cfg['num_epochs'])
        return Flow(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])
    return train_trial(makeTrainer,strict)

if __name__=='__main__':
    Trial = simpleFlowTrial(strict=True)
    Trial({'num_epochs':100,'net_config': {'sigma':.5,'k':32},})