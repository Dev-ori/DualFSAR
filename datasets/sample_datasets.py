import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import decord
from decord import VideoReader, cpu
import numpy as np
import warnings
from .build import dataset_register
from torchvision import transforms

class SampleDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform
        pass

    def __len__(self):
        return 100
    
    def __getitem__(self, idx):
        x = torch.rand(10)
        y = torch.rand(5)
        return x, y

@dataset_register('sample dataset')
def sample_dataset(cfg, mode, global_rank, world_size):
    transform = transforms.Compose([])
    dataset = SampleDataset(transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank,shuffle = True)
    dataloader = DataLoader(dataset, batch_size=cfg['BATCH_SIZE'], num_workers=cfg['WORKER'], shuffle=False, sampler=sampler)

    return dataset, dataloader