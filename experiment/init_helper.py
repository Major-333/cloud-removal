from typing import Optional, Dict, Tuple
import logging
import os
import yaml
import wandb
import torch
from torch import nn, tensor, optim, distributed as dist, multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from dataset.dsen2cr_dataset import Dsen2crDataset
from dataset.basic_dataloader import Seasons
from models.restormer import Restormer
from models.dsen2cr import DSen2_CR
from models.test_model import TestModel
from models.mprnet import MPRNet
from warmup_scheduler import GradualWarmupScheduler
from dataset.visualize import visualize_output_with_groundtruth, visualize_output_with_groundtruth_only_rgb, get_output_with_groundtruth_distribution_by_channel
from utils import init_weights, init_dsen2cr, init_mprnet, init_restormer, init_test_model
from loss.charbonnier_loss import CharbonnierLoss


LOSS_MAPPER = {
    'MSE': torch.nn.MSELoss(),
    'CharbonnierLoss': CharbonnierLoss()
}

MODEL_MAPPER = {
    'MPRNet': init_mprnet,
    'Restormer': init_restormer,
    'DSen2CR': init_dsen2cr,
    'Test': init_test_model
}

def init_wandb(rank: int, group: str) -> Dict:
    job_type = f'rank:{rank}'
    wandb.init(project='cloud removal', group=group, job_type=job_type)
    config = wandb.config
    logging.info(f'config is:{config}')
    return config

class InitHelper(object):
    def __init__(self, config: Dict) -> None:
        self.config = config

    def init_train_dsen2cr_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        scene_black_list = self.config['scene_black_list']
        scene_white_list = self.config['scene_white_list']
        dataset = Dsen2crDataset(self.config['train_dir'],
                                    Seasons(self.config['season']),
                                    scene_white_list=scene_white_list,
                                    scene_black_list=scene_black_list)
        n_val = int(len(dataset) * self.config['val_percent'])
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(1999))
        train_loader = DataLoader(train_set, batch_size=self.config['batch_size'], num_workers=8)
        val_loader = DataLoader(val_set, batch_size=self.config['batch_size'], num_workers=8)
        return train_loader, val_loader

    def init_dsen2cr_dataloader(self, data_dir: str) -> DataLoader:
        scene_black_list = self.config['scene_black_list']
        scene_white_list = self.config['scene_white_list']
        dataset = Dsen2crDataset(data_dir,
                                    Seasons(self.config['season']),
                                    scene_white_list=scene_white_list,
                                    scene_black_list=scene_black_list)
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], num_workers=8)
        return dataloader

    def init_optimizer(self, model: nn.Module) -> Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.config['lr'])

    def get_warmup_scheduler(self, optimizer: Optimizer) -> object:
        min_lr, total_epochs = self.config['min_lr'], self.config['epochs']
        warmup_epochs = 3
        # 3-500 完成一次余弦
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=min_lr)
        # 3 warmuo
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()
        return scheduler

    def get_loss_fn(self):
        return LOSS_MAPPER[self.config['loss_fn']]
    
    def init_model(self) -> nn.Module:
        model = MODEL_MAPPER[self.config['model']]()
        logging.info(f'===== using model: {type(model)} =====')
        model = model.cuda()
        return model


class DDPInitHelper(InitHelper):
    def __init__(self, config: Dict, rank: int, world_size: int) -> None:
        super().__init__(config)
        self.config = config
        self.rank = rank
        self.world_size = world_size
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    def init_ddp(self) -> None:
        dist.init_process_group("gloo", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.rank)

    def init_train_dsen2cr_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        scene_black_list = self.config['scene_black_list']
        scene_white_list = self.config['scene_white_list']
        dataset = Dsen2crDataset(self.config['train_dir'],
                                    Seasons(self.config['season']),
                                    scene_white_list=scene_white_list,
                                    scene_black_list=scene_black_list)
        n_val = int(len(dataset) * self.config['val_percent'])
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(1999))
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set)
        train_loader = DataLoader(train_set, batch_size=self.config['batch_size'], num_workers=8, sampler=train_sampler)
        val_loader = DataLoader(val_set, batch_size=self.config['batch_size'], num_workers=8, sampler=val_sampler)
        return train_loader, val_loader

    def init_model(self) -> nn.Module:
        model = super().init_model()
        return DDP(model, device_ids=[self.rank])