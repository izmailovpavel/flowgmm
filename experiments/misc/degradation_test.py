import os
from oil.datasetup.datasets import CIFAR10,CIFAR100
from oil.model_trainers.classifier import Classifier,simpleClassifierTrial
# from resnets import SplitODEResnet,ODEResnet,LongResnet,RNNBottle
# from resnets import SmallResnet,RNNResnet
from invertible.iEluNetwork import DegredationTester
from oil.tuning.study import Study, train_trial
# from oil.tuning.configGenerator import uniform,logUniform

# log_dir_base = os.path.expanduser('~/tb-experiments/degradation_test')
# cfg_spec = {
#     'dataset': CIFAR10,
#     'network': [DegredationTester],
#     'net_config': [ {},
#                     {'circ':True},
#                     {'circ':True,'slrelu':True},
#                     {'circ':True,'slrelu':True,'ds':'checkerboard'},
#                     {'circ':True,'slrelu':True,'ds':'nn'},
#                     {'circ':True,'slrelu':True,'ds':'avg'},
#                     ],#k=128
#     'loader_config': {'amnt_dev':0,'lab_BS':50},
#     'opt_config':{'lr':.1},
#     'num_epochs':100, 
#     'trainer_config':{'log_dir':lambda cfg:log_dir_base+\
#         '/{}/{}'.format(cfg['dataset'],cfg['network'])}
#     }
#'log_dir':lambda cfg:f'{log_dir_base}/{cfg['dataset']}/{cfg['network']}/s{cfg['net_config']['sigma']}'
#ODEResnet,RNNResnet,,SplitODEResnet,SmallResnet,BezierRNNSplit,BezierODE,BezierRNN


log_dir_base = os.path.expanduser('~/tb-experiments/lrelu_test_slrelu2')
cfg_spec = {
    'dataset': CIFAR10,
    'network': [DegredationTester],
    'net_config':{'circ':True,'slrelu':True},#k=128
    'loader_config': {'amnt_dev':0,'lab_BS':50},
    'opt_config':{'lr':.1},
    'num_epochs':100, 
    'trainer_config':{'log_dir':lambda cfg:log_dir_base+\
        '/{}/{}'.format(cfg['dataset'],cfg['network'])}
    }


do_trial = simpleClassifierTrial(strict=True)
ode_study = Study(do_trial,cfg_spec)
ode_study.run()

