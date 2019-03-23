import os
from oil.datasetup.datasets import CIFAR10,CIFAR100
from oil.model_trainers.cGan import simpleGanTrial
from oil.tuning.study import Study, train_trial
from oil.tuning.configGenerator import uniform,logUniform


log_dir_base = os.path.expanduser('~/tb-experiments/gans')
cfg_spec = {
    'dataset': CIFAR10,
    'num_epochs':100, 
    'trainer_config':{'log_dir':lambda cfg: f'{log_dir_base}/{cfg['dataset'])}'},
    }
Trial = simpleGanTrial(strict=True)


gan_study = Study(Trial, slurm_cfg={'time':'16:00:00'})
gan_study.run(num_trials=1,max_workers=1,new_config_spec=cfg_spec)