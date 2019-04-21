import os
from oil.datasetup.datasets import CIFAR10,CIFAR100
from oil.model_trainers.classifier import Classifier,simpleClassifierTrial
# from resnets import SplitODEResnet,ODEResnet,LongResnet,RNNBottle
# from resnets import SmallResnet,RNNResnet
# from resnets import BezierRNN,BezierODE,BezierRNNSplit
from iresnet import iResnet,iResnetLarge,iResnetLargeV2
from oil.tuning.study import Study, train_trial
# from oil.tuning.configGenerator import uniform,logUniform
from flow import simpleFlowTrial
from iEluNetwork import iEluNet,iEluNetMultiScale,iEluNetMultiScaleLarger
log_dir_base = os.path.expanduser('~/tb-experiments/elu_flow_2e')
cfg_spec = {
    'dataset': [CIFAR10],
    'network': [iEluNetMultiScaleLarger],
    'net_config': {'k':32},
    'loader_config': {'amnt_dev':5000,'lab_BS':64},
    'opt_config':{'lr':.001},
    'num_epochs':5*(10,), 
    'trainer_config':{'log_dir':lambda cfg:log_dir_base+\
        '/{}/{}'.format(cfg['dataset'],cfg['network'])}
    }
#'log_dir':lambda cfg:f'{log_dir_base}/{cfg['dataset']}/{cfg['network']}/s{cfg['net_config']['sigma']}'
#ODEResnet,RNNResnet,,SplitODEResnet,SmallResnet,BezierRNNSplit,BezierODE,BezierRNN
do_trial = simpleFlowTrial(strict=True)
ode_study = Study(do_trial,cfg_spec,study_name='iresnet_circ')
ode_study.run()

lr_search_cfg_spec = {
    'dataset': [CIFAR10],
    'network': [iEluNetMultiScale],
    'net_config': {'k':32},
    'loader_config': {'amnt_dev':5000,'lab_BS':64},
    'opt_config':{'lr':[.003,.001,.0003]},
    'num_epochs':25, 
    'trainer_config':{'log_dir':lambda cfg:log_dir_base+\
        '/{}/{}'.format(cfg['dataset'],cfg['network'])}
    }

ode_study.run(new_config_spec = lr_search_cfg_spec)