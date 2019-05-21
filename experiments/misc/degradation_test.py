import os
import torch
import torch.nn as nn
from oil.datasetup.datasets import CIFAR10,CIFAR100
from oil.model_trainers.classifier import Classifier,simpleClassifierTrial
from flow_ssl.flow_trainer import simpleiClassifierTrial
# from resnets import SplitODEResnet,ODEResnet,LongResnet,RNNBottle
# from resnets import SmallResnet,RNNResnet
from oil.tuning.study import Study, train_trial
from oil.utils.utils import Named,Expression
from flow_ssl.invertible import iConv2d,iSLReLU,SqueezeLayer,ClippediConv2d,iSequential,ActNorm,Id,RandomPadChannels
from flow_ssl.invertible import SqueezeLayer,padChannels,keepChannels,NNdownsample,iAvgPool2d,Flatten,iLeakyReLU
from flow_ssl.invertible import iLogits, iBN, MeanOnlyBN, passThrough, addZslot, Join, pad_circular_nd
from flow_ssl.icnn.icnn import StandardNormal,FlowNetwork,MultiScaleiCNNv2
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

class DegredationTester(FlowNetwork):
    def __init__(self, num_classes=10,k=64,circ=False,slrelu=False,lrelu=None,bn=True,clip=(.01,None),ds='max'):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        
        conv = lambda c1,c2: iConv2d(c1,c2,circ=circ)#ClippediConv2d(c1,c2,clip=clip,circ=circ)#iConv2d(c1,c2,circ=circ)#ClippediConv2d(c1,c2,clip=clip,circ=circ)#iConv2d(c1,c2,circ=circ)
        BN = iBN if bn else lambda c: Id()#ActNorm##lambda c: nn.BatchNorm2d(c)#Expression(lambda x:x)#nn.BatchNorm2d(c)#Expression(lambda x:x)#nn.Sequential()#nn.BatchNorm2d#ActNorm#nn.BatchNorm2d
        relu = iSLReLU if slrelu else nn.ReLU
        if lrelu is not None: relu = lambda: iLeakyReLU(lrelu)
        if ds=='max': downsample = lambda: nn.MaxPool2d(2)
        elif ds=='checkerboard': downsample = lambda: SqueezeLayer(2)
        elif ds=='nn': downsample = lambda: NNdownsample()
        elif ds=='avg': downsample = lambda: iAvgPool2d()
        else: assert False, "unknown option"
        CBR = lambda c1,c2: iSequential(conv(c1,c2),BN(c2),relu())
        self.body = iSequential(
            RandomPadChannels(k-3),
            addZslot(),
            passThrough(*[CBR(k,k) for _ in range(3)]),
            passThrough(downsample()),
            keepChannels(2*k),
            passThrough(*[CBR(2*k,2*k) for _ in range(3)]),
            passThrough(downsample()),
            keepChannels(2*k),
            passThrough(*[CBR(2*k,2*k) for _ in range(3)]),
            Join()
        )
        self.classifier_head = nn.Sequential(
            Expression(lambda z:z[-1]),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.BatchNorm1d(2*k),
            nn.Linear(2*k,num_classes)
        )
        self.flow = iSequential(iLogits(),self.body,Flatten())
        self.prior = StandardNormal(k*32*32)
    


# log_dir_base = os.path.expanduser('~/tb-experiments/lrelu_test_logdet_invertible_nobn_slrelu_randompad')
# cfg_spec = {
#     'dataset': CIFAR10,
#     'network': [DegredationTester],
#     'net_config':{'circ':True,'slrelu':True,'clip':(0.1,1),'ds':'nn','bn':False,'k':64},#k=128
#     'loader_config': {'amnt_dev':0,'lab_BS':50},
#     'opt_config':{'lr':.1},
#     'num_epochs':3*(1,), 
#     'trainer_config':{'log_dir':lambda cfg:log_dir_base+\
#         '/{}/{}'.format(cfg['dataset'],cfg['network']),'ld_weight':.001}
#     }


log_dir_base = os.path.expanduser('~/tb-experiments/multiscale_icnnv2_ld')
cfg_spec = {
    'dataset': CIFAR10,
    'network': [MultiScaleiCNNv2],
    'net_config':{'k':64},#k=128
    'loader_config': {'amnt_dev':0,'lab_BS':50},
    'opt_config':{'lr':.0003},
    'num_epochs':3*(1,), 
    'trainer_config':{'log_dir':lambda cfg:log_dir_base+\
        '/{}/{}'.format(cfg['dataset'],cfg['network']),'ld_weight':.01}
    }

do_trial = simpleiClassifierTrial(strict=True)
ode_study = Study(do_trial,cfg_spec)
ode_study.run()

