import torch

class SequentialWithAutoInverseAndLogDet(torch.nn.Sequential):

    def inverse(self,y):
        j = len(self._modules.values())
        for module in reversed(self._modules.values()):
            j -=1
            #print(f"Inverting layer{j} with module {module}")
            assert hasattr(module,'inverse'), '{} has no inverse defined'.format(module)
            y = module.inverse(y)
        return y

    def logdet(self):
        log_det = 0
        for module in self._modules.values():
            #print(module)
            assert hasattr(module,'logdet'), '{} has no logdet defined'.format(module)
            log_det += module.logdet()
        return log_det

#torch.nn.Sequential = SequentialWithAutoInverseAndLogDet
iSequential = SequentialWithAutoInverseAndLogDet
