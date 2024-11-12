import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import numpy as np
import random
from utils.seed import random_seed

def setup(local_rank, global_rank, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    dist.init_process_group(args.backend, 
                            rank=global_rank, 
                            world_size=args.world_size)
    
    torch.cuda.set_device(local_rank)
    torch.cuda.synchronize()
    
    random_seed(args.seed)
    
def cleanup():
    dist.destroy_process_group()
    
def run_func(idx, func, args):
        local_rank = idx
        global_rank = args.global_rank + idx
        
        print(f"Running basic DDP example on global rank {global_rank}, local rank {local_rank}.")
        print(f"World size : {args.world_size}")
        
        setup(local_rank, global_rank, args)

        func(local_rank, global_rank, args)
        
        cleanup()


def DDP_run(func, args):
    try:
        mp.spawn(run_func,
                args=(func, args, ),
                nprocs = args.local_gpus,
                join=True)
    except KeyboardInterrupt:
        try:
            cleanup()
        except:
            os.system('kill $(ps aux | grep "multiprocessing.spawn" | grep -v grep | awk "{print $2}" )')

def reduce_loss(device, all_loss):
    all_loss = torch.tensor(all_loss).to(device)
    all_loss = torch.mean(all_loss)
    
    dist.reduce(all_loss, dst=0, op=dist.ReduceOp.SUM)
    
    if dist.get_rank() == 0:
        all_loss /= dist.get_world_size()
        all_loss = all_loss.detach().cpu().item()

    return all_loss