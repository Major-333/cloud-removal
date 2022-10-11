import os
import sys
import wandb
import logging
import argparse
import traceback
from tqdm import tqdm
from typing import Optional, Dict, List
import torch
from torch import nn, optim, index_copy, distributed as dist, multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from warmup_scheduler import GradualWarmupScheduler
from datetime import datetime, timedelta
from sen12ms_cr_dataset.build import build_distributed_loaders
from models.build import build_distributed_model
from loss.build import build_loss_fn
from train import Trainer, CHECKPOINT_NAME_PREFIX

class DistributedTrainer(Trainer):
    def __init__(self, config: Dict, local_rank: int, checkpoint_path: Optional[str] = None) -> None:
        self.rank = local_rank
        # initialize PyTorch distributed using environment variables (you could also do this more explicitly by specifying `rank` and `world_size`,
        #  but I find using environment variables makes it so that you can easily use the same script on different machines)
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(f'cuda:{local_rank}')
        self._parse_config(config)
        self.train_loader, self.val_loader, self.test_loader = build_distributed_loaders(self.dataset_path, self.batch_size, self.dataset_file_extension)
        self.model = build_distributed_model(self.model_name, gpu_id=local_rank)
        self.loss_fn = build_loss_fn(self.loss_name)
        self.optimizer = self._get_optimizer(self.model)
        self.scheduler = self._get_scheduler(self.optimizer)
        # for model resum
        self.checkpoint_path = checkpoint_path
        if self.is_resume:
            self.resume_epoch_num = int(self.checkpoint_path.split(CHECKPOINT_NAME_PREFIX)[1])
        # for summary
        self.best_val_psnr = 0
        self.best_val_ssim = 0  


if __name__ == '__main__':
    logging.basicConfig(filename='train.log',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    if not os.getenv('CUDA_VISIBLE_DEVICES'):
        raise ValueError(f'set the env: `CUDA_VISIBLE_DEVICES` first')
    if not os.getenv('LOCAL_RANK'): 
        raise ValueError(f'set the env: `LOCAL_RANK` first')
    if not os.getenv('WANDB_GROUP'):
        raise ValueError(f'set the env: `WANDB_GROUP` first')
    rank =  int(os.getenv('LOCAL_RANK'))
    group = os.getenv('WANDB_GROUP')
    # Initialize wandb
    wandb.init(project='cloud removal V2', group=group, job_type='DDP mode')
    config = wandb.config
    # Initialize run
    trainer = DistributedTrainer(wandb.config, rank)
    trainer.train()