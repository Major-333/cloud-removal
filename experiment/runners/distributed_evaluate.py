import os
import sys
import wandb
import logging
import argparse
import traceback
import pandas as pd
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
from runners.evaluate import Evaluater, EvaluateType, EVAL_SUBDIR_NAME, METRIC_FILENAME, CONFIG_FILEPATH
from utils import setup_seed, get_rois_from_split_file, DEFAULT_LOG_FILENAME, config_logging, parse_wandb_yaml


class DistributedEvaluater(Evaluater):
    def __init__(self, config: Dict, local_rank: int, save_predict: bool=False) -> None:
        # initialize PyTorch distributed using environment variables (you could also do this more explicitly by specifying `rank` and `world_size`,
        #  but I find using environment variables makes it so that you can easily use the same script on different machines)
        dist.init_process_group(backend='nccl', init_method='env://')
        self.local_rank = local_rank
        torch.cuda.set_device(f'cuda:{local_rank}')
        # Load config to trainer
        self._parse_config(config)
        self.config = config
        # Fix random seed for reproducibility
        setup_seed(self.seed)
        # for save.
        self.save_predict = save_predict
        if self.is_master:
            self.save_dir = self._get_save_dir(EVAL_SUBDIR_NAME)
            dist.broadcast_object_list([self.save_dir])
        else:
            broadcast_msg = [None]
            dist.broadcast_object_list(broadcast_msg, src=0)
            self.save_dir = broadcast_msg[0]
        # init logging
        logging_file_path = os.path.join(self.save_dir, DEFAULT_LOG_FILENAME)
        config_logging(filename=logging_file_path)
        logging.info(f'rank:{self.local_rank} Evaluater has been initialized. save predict is: {self.save_predict}')
        # Init dataloader
        logging.info(f'use spit file:{self.split_file_path}')
        _, val_rois, test_rois = get_rois_from_split_file(self.split_file_path)
        self.val_loader = build_distributed_loaders_with_rois(self.dataset_path, self.batch_size, self.dataset_file_extension, val_rois, debug=self.debug, return_with_triplet=save_predict)
        self.test_loader = build_distributed_loaders_with_rois(self.dataset_path, self.batch_size, self.dataset_file_extension, test_rois, debug=self.debug, return_with_triplet=save_predict)
        # Init model
        self.model = build_distributed_model(self.model_name, gpu_id=local_rank)
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        state = torch.load(checkpoint_path)
        self.model.module.load_state_dict(state)

if __name__ == '__main__':
    if not os.getenv('CUDA_VISIBLE_DEVICES'):
        raise ValueError(f'set the env: `CUDA_VISIBLE_DEVICES` first')
    if not os.getenv('CHECKPOINT_PATH'):
        raise ValueError(f'set the env: `CHECKPOINT_PATH` first')
    if not os.getenv('SAVE_PREDICT'):
        raise ValueError(f'set the env: `SAVE_PREDICT` first')
    config = parse_wandb_yaml(CONFIG_FILEPATH)
    rank =  int(os.getenv('LOCAL_RANK'))
    checkpoint_path = os.getenv('CHECKPOINT_PATH')
    save_predict = True if os.getenv('SAVE_PREDICT') == "1" else False
    print(f'config:\n{config}')
    print(f'checkpoint_path:\n{checkpoint_path}')
    # Initialize evaluater
    evaluater = DistributedEvaluater(config=config, local_rank=rank, save_predict=save_predict)
    evaluater.evaluate_checkpoint(checkpoint_path, eval_type=EvaluateType.TEST)
