import glob
import io
from pathlib import Path
from torch import nn
import os
import torch
from collections import OrderedDict
from timm.utils import get_state_dict

def save_model(args, epoch, model, optimizer, loss_scaler, save_dir, save_file_name, model_ema):
    save_dict = {
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'epoch' : epoch,
        'args' : args,
    }
    
    if loss_scaler is not None:
        save_dict['scaler'] = loss_scaler.state_dict()

    if model_ema is not None:
        save_dict['model_ema'] = get_state_dict(model_ema)
        
    torch.save(save_dict, os.path.join(save_dir, f"{save_file_name}.pth"))

def load_model(checkpoint, model, model_keys):
    ckpt_state_dict = None

    for model_key in model_keys.split('|'):
        if model_key in checkpoint:
            ckpt_state_dict = checkpoint[model_key]
            break
        
    if ckpt_state_dict is None:
        ckpt_state_dict = checkpoint
            
    model.load_state_dict(ckpt_state_dict)
    
def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)
   
    
def auto_load_model(args, model, model_ema, optimizer, loss_scaler, logger=None):
    output_dir = Path(args.output_dir)
    if args.auto_resume and len(args.checkpoint) == 0:
        
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.checkpoint = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            if logger is not None:
                logger.log(logger.INFO, "Auto resume checkpoint: %s" % args.checkpoint)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        load_model(checkpoint, model, args.model_keys)
        
        logger.log(logger.INFO,"checkpoint %s" % args.checkpoint)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if hasattr(args, 'model_ema') and args.model_ema:
                _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            logger.log(logger.INFO,"With optim & sched!")