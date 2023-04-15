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
from sen12ms_cr_dataset.build import build_distributed_loaders, build_distributed_loaders_with_rois
from models.build import build_distributed_model
from loss.build import build_loss_fn
from runners.train import Trainer, CHECKPOINT_NAME_PREFIX, TRAIN_SUBDIR_NAME
from runners.evaluate import Evaluater, EvaluateType
from utils import setup_seed, get_rois_from_split_file, DEFAULT_LOG_FILENAME, config_logging

class DistributedTrainer(Trainer):
    def __init__(self, config: Dict, local_rank: int, checkpoint_path: Optional[str] = None) -> None:
        self._runner_init(config, TRAIN_SUBDIR_NAME)
        self._init_distributed(local_rank)
        self._init_dataloader()
        self._init_model()
        self._init_optim()
        self._prepare_model_resum(checkpoint_path)
        self._prepare_summary()
    
    def _init_distributed(self, local_rank):
         # initialize PyTorch distributed using environment variables (you could also do this more explicitly by specifying `rank` and `world_size`,
        #  but I find using environment variables makes it so that you can easily use the same script on different machines)
        dist.init_process_group(backend='nccl', init_method='env://')
        self.local_rank = local_rank
        torch.cuda.set_device(f'cuda:{local_rank}')

    def _init_dataloader(self):
        if self.split_file_path:
            logging.info(f'use spit file:{self.split_file_path}')
            train_rois, val_rois, _ = get_rois_from_split_file(self.split_file_path)
            self.train_loader = build_distributed_loaders_with_rois(self.dataset_path, self.batch_size, self.dataset_file_extension, train_rois, use_cloud_mask=self.use_cloud_mask, debug=False)
            self.val_loader = build_distributed_loaders_with_rois(self.dataset_path, self.batch_size, self.dataset_file_extension, val_rois, use_cloud_mask=self.use_cloud_mask, debug=self.debug)
        else:
            logging.info(f'using random split')
            self.train_loader, self.val_loader, _ = build_distributed_loaders(self.dataset_path, self.batch_size, self.dataset_file_extension)
    
    def _init_model(self):
        self.model = build_distributed_model(self.model_name, gpu_id=self.local_rank)
        self.loss_fn = build_loss_fn(self.loss_name)

    def _init_optim(self):
        self.optimizer = self._get_optimizer(self.model)
        self.scheduler = self._get_scheduler(self.optimizer)

    def _prepare_model_resum(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        if self.is_resume:
            self.resume_epoch_num = int(self.checkpoint_path.split(CHECKPOINT_NAME_PREFIX)[1])

    def _prepare_summary(self):
        self.best_val_psnr = 0
        self.best_val_ssim = 0

    @property
    def is_master(self) -> bool:
        return self.local_rank == 0

    def train(self) -> None:
        for epoch in range(1, self.max_epoch + 1):
            self.model.train()
            epoch_loss = 0.0
            training_info = {'learning rate': self.optimizer.param_groups[0]['lr']}
            if not self.is_resume or epoch > self.resume_epoch_num:
                # let all processes sync up before starting with a new epoch of training
                dist.barrier()
                for index, data_batch in enumerate(tqdm(self.train_loader, desc='Epoch: {}'.format(epoch))):
                    self.optimizer.zero_grad()
                    cloudy, ground_truth = data_batch
                    cloudy, ground_truth = cloudy.cuda(), ground_truth.cuda()
                    output = self.model(cloudy)
                    loss = self.loss_fn(output, ground_truth)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss
                training_info = {**training_info , **{'epoch_loss': epoch_loss.item()}}
                if epoch % self.validate_every == 0:
                    # let all processes sync up before starting with a new epoch of validating
                    dist.barrier()
                    self.model.eval()
                    metric = Evaluater.evaluate(lambda x: self.model(x), self.val_loader, EvaluateType.VALIDATE)
                    training_info = {**training_info, **metric}
                    logging.info(f'metric:{metric}')
                    is_update = self._update_summary(metric, epoch, metric_prefix=EvaluateType.VALIDATE.value)
                    if is_update and self.local_rank == 0:
                        self._save(self.model, epoch)
            self._logs(training_info)
            self.scheduler.step()
        self._finish()

class DistributedTrainStarter():
    def __init__(self) -> None:
        if not os.getenv('CUDA_VISIBLE_DEVICES'):
            raise ValueError(f'set the env: `CUDA_VISIBLE_DEVICES` first')
        if not os.getenv('LOCAL_RANK'): 
            raise ValueError(f'set the env: `LOCAL_RANK` first')
        if not os.getenv('WANDB_GROUP'):
            raise ValueError(f'set the env: `WANDB_GROUP` first')
        group = os.getenv('WANDB_GROUP')
        wandb.init(project='cloud removal V2', group=group, job_type='DDP mode')
        rank = int(os.getenv('LOCAL_RANK'))
        self.init_trainner(rank)

    def init_trainner(self, rank):
        self.trainer = DistributedTrainer(wandb.config, rank)

    def start_train(self):
        self.trainer.train()

if __name__ == '__main__':
    train_starter = DistributedTrainStarter()
    train_starter.start_train()