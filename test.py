import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import tqdm

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size = world_size)

def cleanup():
    dist.destroy_process_group()
    
def demo_basic(rank, world_size=4):
    print(f"Running basic DDP example on rank {rank}.")
    print(f"World size : {world_size}")
    setup(rank, world_size)
    
    cleanup()

    

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             nprocs = 4,
             join=True)
    
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()

    world_size = n_gpus

    run_demo(demo_basic, world_size)
