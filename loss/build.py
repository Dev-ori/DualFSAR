from torch import nn

loss_registry = {}

def loss_register(name=None):
    def decorator(obj):
        loss_registry[name or obj.__name__] = obj
        return obj
    return decorator

def create_criterion(args):
    loss = args.loss.lower()
    loss = loss.replace('_', ' ').replace('loss', '').strip()

    criterion = loss_registry.get(loss, None)
        
    if criterion is None:
        raise Exception(f"'loss' doesn't exist. loss's input must be {', '.join(loss_registry.keys())}")
    
    return criterion()