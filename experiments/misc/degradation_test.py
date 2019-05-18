import os
import torch
import torch.nn as nn
from oil.datasetup.datasets import CIFAR10,CIFAR100
from oil.model_trainers.classifier import Classifier,simpleClassifierTrial
# from resnets import SplitODEResnet,ODEResnet,LongResnet,RNNBottle
# from resnets import SmallResnet,RNNResnet
from oil.tuning.study import Study, train_trial
from oil.utils.utils import Named,Expression
from flow_ssl.invertible import iConv2d,iSLReLU,SqueezeLayer,NNdownsample,iAvgPool2d,ClippediConv2d,iSequential
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

class DegredationTester(nn.Module,metaclass=Named):
    def __init__(self, num_classes=10,k=128,circ=False,slrelu=False,lrelu=None,clip=(.01,None),ds='max'):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        
        conv = lambda c1,c2: ClippediConv2d(c1,c2,clip=clip,circ=circ)#iConv2d(c1,c2,circ=circ)
        BN = nn.BatchNorm2d
        relu = iSLReLU if slrelu else nn.ReLU
        if lrelu is not None: relu = lambda: nn.LeakyReLU(lrelu)
        if ds=='max': downsample = nn.MaxPool2d(2)
        elif ds=='checkerboard': downsample = SqueezeLayer(2)
        elif ds=='nn': downsample = NNdownsample()
        elif ds=='avg': downsample = iAvgPool2d()
        else: assert False, "unknown option"
        CBR = lambda c1,c2: nn.Sequential(conv(c1,c2),BN(c2),relu())
        self.net = nn.Sequential(
            CBR(3,k),
            CBR(k,k),
            CBR(k,2*k),
            downsample,
            Expression(lambda x: x[:,:2*k]),
            CBR(2*k,2*k),
            CBR(2*k,2*k),
            CBR(2*k,2*k),
            downsample,
            Expression(lambda x: x[:,:2*k]),
            CBR(2*k,2*k),
            CBR(2*k,2*k),
            CBR(2*k,2*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(2*k,num_classes)
        )
    def forward(self,x):
        return self.net(x)



log_dir_base = os.path.expanduser('~/tb-experiments/lrelu_test_slrelu2_2clip')
cfg_spec = {
    'dataset': CIFAR10,
    'network': [DegredationTester],
    'net_config':{'circ':True,'slrelu':False,'clip':(0.01,None)},#k=128
    'loader_config': {'amnt_dev':0,'lab_BS':50},
    'opt_config':{'lr':.1},
    'num_epochs':100, 
    'trainer_config':{'log_dir':lambda cfg:log_dir_base+\
        '/{}/{}'.format(cfg['dataset'],cfg['network'])}
    }


do_trial = simpleClassifierTrial(strict=True)
ode_study = Study(do_trial,cfg_spec)
ode_study.run()

