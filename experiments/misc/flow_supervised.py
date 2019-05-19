import os
from oil.datasetup.datasets import CIFAR10,CIFAR100
from oil.model_trainers.classifier import Classifier,simpleClassifierTrial
from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, recursively_update,islice
from oil.tuning.study import train_trial
from oil.datasetup.dataloaders import getLabLoader
import collections
import torch
# from resnets import SplitODEResnet,ODEResnet,LongResnet,RNNBottle
# from resnets import SmallResnet,RNNResnet
# from resnets import BezierRNN,BezierODE,BezierRNNSplit
#from iresnet import iResnet,iResnetLarge,iResnetLargeV2
from oil.tuning.study import Study, train_trial
from oil.architectures.img_classifiers import smallCNN,layer13s
# from oil.tuning.configGenerator import uniform,logUniform
from flow_ssl.flow_trainer import simpleFlowTrial
from flow_ssl.icnn.icnn import iCNN3d2,iCNN3d
log_dir_base = os.path.expanduser('~/tb-experiments/icnn3d_retrial3')
cfg_spec = {
    'dataset': CIFAR10,
    'network': iCNN3d,
    'net_config': {},
    'loader_config': {'amnt_dev':5000,'lab_BS':50},
    'opt_config':{'lr':.001},
    'num_epochs':5, 
    'trainer_config':{'log_dir':lambda cfg:log_dir_base+\
        '/{}/{}'.format(cfg['dataset'],cfg['network'])}
    }
#'log_dir':lambda cfg:f'{log_dir_base}/{cfg['dataset']}/{cfg['network']}/s{cfg['net_config']['sigma']}'
#ODEResnet,RNNResnet,,SplitODEResnet,SmallResnet,BezierRNNSplit,BezierODE,BezierRNN

def classificationTrial(strict=False):
    def makeTrainer(config):
        cfg = {
            'dataset': CIFAR10,'network':layer13s,'net_config': {},
            'loader_config': {'amnt_dev':5000,'lab_BS':50, 'pin_memory':True,'num_workers':2},
            'opt_config':{'lr':1e-3},
            'num_epochs':100,'trainer_config':{},
            }
        recursively_update(cfg,config)
        trainset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']))
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
        opt_constr = lambda params: torch.optim.Adam(params, **cfg['opt_config'])
        lr_sched = cosLr(cfg['num_epochs'])
        return Classifier(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])
    return train_trial(makeTrainer,strict)

do_trial = classificationTrial(strict=True)
ode_study = Study(do_trial,cfg_spec,study_name='supervised')
ode_study.run()