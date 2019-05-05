import os
from oil.datasetup.datasets import CIFAR10,CIFAR100
from oil.model_trainers.classifier import Classifier,simpleClassifierTrial
# from resnets import SplitODEResnet,ODEResnet,LongResnet,RNNBottle
# from resnets import SmallResnet,RNNResnet
from iEluNetwork import iEluNet
from oil.tuning.study import Study, train_trial
# from oil.tuning.configGenerator import uniform,logUniform

log_dir_base = os.path.expanduser('~/tb-experiments/ielu50_iSLReLU')
cfg_spec = {
    'dataset': CIFAR10,
    'network': iEluNet,
    'net_config': {'k':32},
    'loader_config': {'amnt_dev':5000,'lab_BS':64},
    'opt_config':{'lr':.1},
    'num_epochs':50, 
    'trainer_config':{'log_dir':lambda cfg:log_dir_base+\
        '/{}/{}'.format(cfg['dataset'],cfg['network'])}
    }
#'log_dir':lambda cfg:f'{log_dir_base}/{cfg['dataset']}/{cfg['network']}/s{cfg['net_config']['sigma']}'
#ODEResnet,RNNResnet,,SplitODEResnet,SmallResnet,BezierRNNSplit,BezierODE,BezierRNN
do_trial = simpleClassifierTrial(strict=True)
ode_study = Study(do_trial,cfg_spec)
ode_study.run()