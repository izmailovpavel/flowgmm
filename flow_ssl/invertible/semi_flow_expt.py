from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, recursively_update,islice
from oil.tuning.study import train_trial
from oil.datasetup.dataloaders import getLabLoader
from oil.datasetup.datasets import CIFAR10
from oil.architectures.img_classifiers import layer13s
from iresnet import iResnet,iResnetLarge
from iEluNetwork import iEluNet,iEluNetMultiScale,iEluNetMultiScaleLarger
from oil.tuning.study import Study, train_trial
from flow_semisupervised import semiFlowTrial
import collections
import os
import copy
# Trial = SemiFlowTrial(strict=True)
# Trial({'num_epochs':100,'net_config': {'sigma':.5,'k':32},})
log_dir_base = os.path.expanduser('~/tb-experiments/elu_semi_flow')
cfg_spec = {
    'dataset': [CIFAR10],
    'network': [iEluNetMultiScaleLarger],
    'net_config': {'k':32},
    'opt_config':{'lr':[.01,.003,.001,.0003]},
    'num_epochs':5, 
    'trainer_config':{'log_dir':lambda cfg:log_dir_base+\
        '/{}/{}'.format(cfg['dataset'],cfg['network']),'unlab_weight':[1.,10,.1]}
    }
#'log_dir':lambda cfg:f'{log_dir_base}/{cfg['dataset']}/{cfg['network']}/s{cfg['net_config']['sigma']}'
#ODEResnet,RNNResnet,,SplitODEResnet,SmallResnet,BezierRNNSplit,BezierODE,BezierRNN
do_trial = semiFlowTrial(strict=True)
ode_study = Study(do_trial,cfg_spec,study_name='semi_flow_hypers')
ode_study.run(10,5)