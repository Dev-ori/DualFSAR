from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.optim import SGD
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class test(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.ModuleDict({
            'abc' : nn.Linear(10, 10)
        })
        
    def forward(x):
        return x
    
class CosineAnnealingLRwithWarmUp(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False, warmup_start_value=0, warmup_epoch=-1, gamma = 1):
        self.warmup_start_value=warmup_start_value
        self.warmup_epoch=warmup_epoch
        self.gamma = gamma
        super().__init__(optimizer=optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch, verbose=verbose)

    def _consine_annealing_get_lr(self):
        return super().get_lr()
    
    def _warmup_get_lr(self):
        return [(base_lr - self.warmup_start_value) * self.last_epoch / self.warmup_epoch + self.warmup_start_value for base_lr in self.base_lrs]
    
    def get_lr(self):
        if self.last_epoch <= self.warmup_epoch:
            print('warmup')
            return self._warmup_get_lr()
        else:
            if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
                self.base_lrs = [base_lr * self.gamma for base_lr in self.base_lrs]
            return self._consine_annealing_get_lr()

class CosineAnnealingWarmRestartsWitchWarmUp(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, warmup_start_value=0, warmup_epoch=-1, gamma = 1):
        self.warmup_start_value=warmup_start_value
        self.warmup_epoch=warmup_epoch
        self.gamma = gamma
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
        
    def _consine_annealing_get_lr(self):
        return super().get_lr()

    def _warmup_get_lr(self):
        return [(base_lr - self.warmup_start_value) * self.last_epoch / self.warmup_epoch + self.warmup_start_value for base_lr in self.base_lrs]
    
    def get_lr(self):
        if self.last_epoch <= self.warmup_epoch:
            return self._warmup_get_lr()
        else:
            if self.T_cur ==0 and self.last_epoch > self.T_0 + self.warmup_epoch:
                self.base_lrs = [base_lr * self.gamma for base_lr in self.base_lrs]
            return self._consine_annealing_get_lr()
        
    def step(self):
        if self.last_epoch <= self.warmup_epoch:
            super(CosineAnnealingWarmRestarts, self).step()
        else :
            super().step()

if __name__ == "__main__":

    model = test()
    optimizer = SGD(model.parameters(),lr = 0.01,momentum=0.99)

    scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=2, eta_min=0)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=0.001)
    # scheduler = CosineAnnealingLRwithWarmUp(optimizer=optimizer, T_max=20, eta_min=0,warmup_epoch=5, gamma = 0.9)
    scheduler = CosineAnnealingWarmRestartsWitchWarmUp(optimizer=optimizer, T_0=20, T_mult=1, eta_min=0, warmup_epoch=5, gamma = 1)

    lr = []
    for i in range(100):
        lr.append(scheduler.get_last_lr()[0])
        scheduler.step()

    plt.plot(lr)
    plt.grid()
    plt.xticks(range(0, 100, 5))
    plt.show()
