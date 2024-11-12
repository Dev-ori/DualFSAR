from torch import nn
from torch import optim
from .build import optimizer_register


@optimizer_register("sgd")
def SGD(parameters, args):
    opt_args = dict(lr=args.lr, 
                    weight_decay=args.weight_decay, 
                    momentum=args.momentum)
    
    if hasattr(args, 'opt_nesterov') and args.opt_nesterov is not None:
        opt_args['nesterov'] = args.opt_nesterov
        
    if hasattr(args, 'opt_dampening') and args.opt_dampening is not None:
        opt_args['dampening'] = args.opt_dampening

    return optim.SGD(parameters, **opt_args)
    
@optimizer_register("adam")
def Adam(parameters, args):
    opt_args = dict(lr=args.opt_lr, 
                    weight_decay=args.weight_decay)
    
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
        
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
        
    return optim.Adam(parameters, **opt_args)


@optimizer_register("adamw")
def Adam(parameters, args):
    opt_args = dict(lr=args.opt_lr, 
                    weight_decay=args.weight_decay)
    
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
        
    return optim.AdamW(parameters, **opt_args)