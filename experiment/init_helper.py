from typing import Optional, Dict, Tuple
import logging
import os
import yaml
import wandb
import torch
from torch import index_copy, nn, tensor, optim, distributed as dist, multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from models.restormer import Restormer
from models.dsen2cr import DSen2_CR
from models.test_model import TestModel
from models.mprnet import MPRNet
from warmup_scheduler import GradualWarmupScheduler
from sen12ms_cr_dataset.visualize import visualize_output_with_groundtruth, visualize_output_with_groundtruth_only_rgb, get_output_with_groundtruth_distribution_by_channel
from utils import init_weights, init_dsen2cr, init_mprnet, init_restormer, init_test_model, init_TSOCR_V1_model, init_TSOCR_V2_model, init_TSOCR_V3_model, init_TSOCR_V1m_model, init_TSOCR_V2m_model
from loss.charbonnier_loss import CharbonnierLoss

LOSS_MAPPER = {'MSE': torch.nn.MSELoss(), 'CharbonnierLoss': CharbonnierLoss()}

MODEL_MAPPER = {
    'MPRNet': init_mprnet,
    'Restormer': init_restormer,
    'DSen2CR': init_dsen2cr,
    'Test': init_test_model,
    'TSOCR_V0': init_restormer,
    'TSOCR_V0.5': init_test_model,
    'TSOCR_V1': init_TSOCR_V1_model,
    'TSOCR_V1m': init_TSOCR_V1m_model,
    'TSOCR_V2': init_TSOCR_V2_model,
    'TSOCR_V2m': init_TSOCR_V1m_model,
    'TSOCR_V3': init_TSOCR_V3_model,
}


def init_wandb(rank: int, group: str) -> Dict:
    job_type = f'rank:{rank}'
    wandb.init(project='cloud removal', group=group, job_type=job_type)
    config = wandb.config
    logging.info(f'config is:{config}')
    return config


def init_wandb_in_dp(group: str) -> Dict:
    wandb.init(project='cloud removal', group=group, job_type='DP mode')
    config = wandb.config
    logging.info(f'config is:{config}')
    return config


class InitHelper(object):

    def __init__(self, config: Dict) -> None:
        self.config = config

    def init_optimizer(self, model: nn.Module) -> Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.config['lr'])

    def get_warmup_scheduler(self, optimizer: Optimizer) -> object:
        min_lr, total_epochs = self.config['min_lr'], self.config['epochs']
        warmup_epochs = 3
        # 3-500 完成一次余弦
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=min_lr)
        # 3 warmuo
        scheduler = GradualWarmupScheduler(optimizer,
                                           multiplier=1,
                                           total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)
        scheduler.step()
        return scheduler