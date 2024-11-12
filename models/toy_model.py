from torch import nn
from timm.models.registry import register_model

@register_model
def toy_model(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,  **kwargs):
    model = ToyModel(in_dim = 10, out_dim= 5, **kwargs)
    
    return model

class ToyModel(nn.Module):
    def __init__(self, in_dim = 10, out_dim = 5):
        super().__init__()
        
        self.layers = nn.ModuleDict({
            'abc' : nn.Linear(in_dim, out_dim)
        })
        
        
    def forward(self, x):
        return self.layers['abc'](x)
    
    
