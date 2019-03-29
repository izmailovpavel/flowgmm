import os
from oil.datasetup.datasets import CIFAR10,CIFAR100
from oil.model_trainers.classifier import Classifier,simpleClassifierTrial
# from resnets import SplitODEResnet,ODEResnet,LongResnet,RNNBottle
# from resnets import SmallResnet,RNNResnet
# from resnets import BezierRNN,BezierODE,BezierRNNSplit
from iresnet import iResnet
from oil.tuning.study import Study, train_trial
# from oil.tuning.configGenerator import uniform,logUniform

log_dir_base = os.path.expanduser('~/tb-experiments/iresnet_singular2')
cfg_spec = {
    'dataset': CIFAR10,
    'network': iResnet,
    'net_config': {'sigma':[.7,.8]},
    'loader_config': {'amnt_dev':5000,'lab_BS':64},
    'opt_config':{'lr':.1},
    'num_epochs':[25,50], 
    'trainer_config':{'log_dir':lambda cfg:log_dir_base+\
        '/{}/{}/s{}'.format(cfg['dataset'],cfg['network'],cfg['net_config']['sigma'])}
    }
#ODEResnet,RNNResnet,,SplitODEResnet,SmallResnet,BezierRNNSplit,BezierODE,BezierRNN
do_trial = simpleClassifierTrial(strict=True)
ode_study = Study(do_trial,cfg_spec)
ode_study.run()