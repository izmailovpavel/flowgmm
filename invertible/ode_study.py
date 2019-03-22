import os
from oil.datasetup.datasets import CIFAR10,CIFAR100
from oil.model_trainers.classifier import Classifier,simpleClassifierTrial
from resnets import SplitODEResnet,ODEResnet,LongResnet,RNNBottle
from resnets import SmallResnet,RNNResnet
from resnets import BezierRNN,BezierODE,BezierRNNSplit
from oil.tuning.study import Study, train_trial
from oil.tuning.configGenerator import uniform,logUniform


log_dir_base = os.path.expanduser('~/tb-experiments/ode_resnets2')
cfgs = ({
    'dataset': CIFAR10,
    'network': net,
    'net_config': {'gn':True},
    'loader_config': {'amnt_dev':5000,'lab_BS':50},
    'num_epochs':100, 
    'trainer_config':{'log_dir':lambda cfg:log_dir_base+\
        '/{}/{}'.format(cfg['dataset'],cfg['network'])}
    } for net in [BezierRNNSplit,BezierODE,BezierRNN])
#ODEResnet,RNNResnet,,SplitODEResnet,SmallResnet,
do_trial = simpleClassifierTrial(strict=True)
ode_study = Study(do_trial, slurm_cfg={'time':'4:00:00'})
for cfg in cfgs:
    ode_study.run(num_trials=1,max_workers=1,new_config_spec=cfg)