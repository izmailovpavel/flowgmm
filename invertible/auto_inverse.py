import torch

class SequentialWithAutoInverseAndLogDet(torch.nn.Sequential):

    def inverse(self,y):
        for module in reversed(self._modules.values()):
            assert hasattr(module,'inverse'), f'{module} has no inverse defined'
            y = module.inverse(y)
        return y

    def logdet(self,x):
        log_det = 0
        for module in self._modules.values():
            assert hasattr(module,'logdet'), f'{module} has no logdet defined'
            log_det += module.logdet(x)
        return log_det

#torch.nn.Sequential = SequentialWithAutoInverseAndLogDet
iSequential = SequentialWithAutoInverseAndLogDet