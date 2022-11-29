import shutil
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
from torch.optim import Optimizer
from warmup_scheduler import GradualWarmupScheduler
from datetime import datetime, timedelta
from sen12ms_cr_dataset.build import build_distributed_loaders_with_rois, build_loaders, build_loaders_with_rois
from models.build import build_model_with_dp
from loss.build import build_loss_fn
from runners.evaluate import EvaluateType, Evaluater
from utils import get_rois_from_split_file, increment_path, setup_seed, config_logging, DEFAULT_LOG_FILENAME
from runners.runner import Runner

START_TIME = (datetime.utcnow() + timedelta(hours=8)).strftime('%Y%m%d%H%M')
CHECKPOINT_NAME_PREFIX = 'Epoch'
TRAIN_SUBDIR_NAME = 'train'
VAL_SUBDIR_NAME = 'val'
WEIGHTS_DIR_NAME = 'weights'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def init_wandb_with_dp(group: str) -> Dict:
    wandb.init(project='cloud removal V2', group=group, job_type='DP mode')
    config = wandb.config
    logging.info(f'config is:{config}')
    return config


class Trainer(Runner):

    def __init__(self, config: Dict, gpus: List[int], checkpoint_path: Optional[str] = None) -> None:
        super(Trainer, self).__init__(config, TRAIN_SUBDIR_NAME)
        # Init dataloader
        train_rois, val_rois, test_rois = get_rois_from_split_file(self.split_file_path)
        self.train_loader = build_loaders_with_rois(self.dataset_path, self.batch_size, self.dataset_file_extension, train_rois, debug=self.debug)
        self.val_loader = build_loaders_with_rois(self.dataset_path, self.batch_size, self.dataset_file_extension, val_rois, debug=self.debug)
        # Init model and optim
        self.gpus = gpus
        self.model = build_model_with_dp(self.model_name, self.gpus)
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

    def _init_runner(self, config: Dict, save_subdir_name: str):
        super(Trainer, self).__init__(config, save_subdir_name)

    def _get_optimizer(self, model: nn.Module) -> Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.lr)

    def _get_scheduler(self, optimizer: Optimizer) -> object:
        min_lr, total_epochs = self.min_lr, self.max_epoch
        warmup_epochs = 3
        # 3-500 完成一次余弦
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=min_lr)
        # 3 warmup
        scheduler = GradualWarmupScheduler(optimizer,
                                           multiplier=1,
                                           total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)
        return scheduler

    def _update_summary(self, metric: Dict, epoch: int, metric_prefix: str) -> bool:
        is_update = False
        if metric[f'{metric_prefix}_psnr'] > self.best_val_psnr:
            self.best_val_psnr = metric[f'{metric_prefix}_psnr']
            self.best_val_psnr_epoch = epoch
            logging.info(f'best val psnr update:{self.best_val_psnr}, on epoch:{epoch}')
            wandb.run.summary['best_val_psnr'] = self.best_val_psnr
            wandb.run.summary['best_val_psnr_epoch'] = epoch
            is_update = True
        if metric[f'{metric_prefix}_ssim'] > self.best_val_ssim:
            self.best_val_ssim = metric[f'{metric_prefix}_ssim']
            self.best_val_ssim_epoch = epoch
            logging.info(f'best val ssim update:{self.best_val_ssim}, on epoch:{epoch}')
            wandb.run.summary['best_val_ssim'] = self.best_val_ssim
            wandb.run.summary['best_val_ssim_epoch'] = epoch
            is_update = True
        return is_update            

    @property
    def is_resume(self) -> bool:
        return self.checkpoint_path is not None

    def _logs(self, traning_info: Dict) -> None:
        wandb.log(traning_info)

    def _finish(self) -> None:
        logging.info(f'=== The training is complete. ===')
        logging.info(f'Best PSNR in validation:{self.best_val_psnr}, at {self.best_val_psnr_epoch}')
        logging.info(f'Best SSIM in validation:{self.best_val_psnr}, at {self.best_val_ssim_epoch}')
        logging.info(f'=== The training is complete. ===')
        wandb.finish()

    def _save(self, model: nn.Module, epoch: int) -> None:
        weights_dir = os.path.join(self.save_dir, WEIGHTS_DIR_NAME)
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        checkpoint_name = f'{CHECKPOINT_NAME_PREFIX}{epoch}.pt'
        file_path = os.path.join(weights_dir, checkpoint_name)
        logging.info(f'Will save model into{file_path}')
        print(f'Will save model into{file_path}')
        torch.save(model.module.state_dict(), file_path)

    def train(self) -> None:
        for epoch in range(1, self.max_epoch + 1):
            self.model.train()
            epoch_loss = 0.0
            training_info = {'learning rate': self.optimizer.param_groups[0]['lr']}
            if not self.is_resume or epoch > self.resume_epoch_num:
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
                    metric = Evaluater.evaluate(self.model, self.val_loader, EvaluateType.VALIDATE)
                    training_info = {**training_info, **metric}
                    is_update = self._update_summary(metric, epoch, metric_prefix=EvaluateType.VALIDATE.value)
                    if is_update:
                        self._save(self.model, epoch)
            self._logs(training_info)
            self.scheduler.step()
        self._finish()

def run(group: str, gpus: List[int], checkpoint_relpath: Optional[str] = None):
    try:
        config = init_wandb_with_dp(group)
        checkpoint_path = None
        if checkpoint_relpath:
            checkpoint_path = os.path.join(config['checkpoints_dir'], checkpoint_relpath)
            logging.info(f'===loading model state from:{checkpoint_path}===')
        trainer = Trainer(config, gpus, checkpoint_path=checkpoint_path)
        trainer.train()
    except Exception as e:
        logging.error(f'Falied in training:{str(e)}, traceback:{traceback.format_exc()}')


def get_args():
    parser = argparse.ArgumentParser(description='Train the Cloud Remove network')
    parser.add_argument('--resume', '-f', type=str, required=False, help='Load model from a .pth file')
    return parser.parse_args()


if __name__ == '__main__':
    if not os.getenv('CUDA_VISIBLE_DEVICES'):
        raise ValueError(f'set the env: `CUDA_VISIBLE_DEVICES` first')
    gpus = [idx for idx in range(len(os.getenv('CUDA_VISIBLE_DEVICES').split(',')))]
    group = f'experiment-{wandb.util.generate_id()}'
    args = get_args()
    if args.resume:
        checkpoint_path = args.resume
        run(group, gpus, checkpoint_path)
    else:
        run(group, gpus)
