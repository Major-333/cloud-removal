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
from sen12ms_cr_dataset.build import build_loaders
from models.build import build_model_with_dp
from loss.build import build_loss_fn
from evaluate import EvaluateType, Evaluater
from utils import increment_path

START_TIME = (datetime.utcnow() + timedelta(hours=8)).strftime('%Y%m%d%H%M')
CHECKPOINT_NAME_PREFIX = 'Epoch'
TRAIN_SUBDIR_NAME = 'train'
VAL_SUBDIR_NAME = 'val'
EXP_SUBDIR_NAME = 'exp'
WEIGHTS_DIR_NAME = 'weights'
CONFIG_NAME = 'config.yaml'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
WANDB_CONFIG = 'config-defaults.yaml'

def init_wandb_with_dp(group: str) -> Dict:
    wandb.init(project='cloud removal V2', group=group, job_type='DP mode')
    config = wandb.config
    logging.info(f'config is:{config}')
    return config


class Trainer(object):

    def __init__(self, config: Dict, gpus: List[int], checkpoint_path: Optional[str] = None) -> None:
        self._parse_config(config)
        self.config = config.copy()
        self.gpus = gpus
        self.train_loader, self.val_loader, self.test_loader = build_loaders(self.dataset_path, self.batch_size,
                                                                             self.dataset_file_extension)
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
        # for save.
        self.train_exp_dir = self._get_train_exp_dir()

    def _get_train_exp_dir(self) -> str:
        exp_dir = os.path.join(self.save_dir, TRAIN_SUBDIR_NAME, EXP_SUBDIR_NAME)
        exp_dir = increment_path(exp_dir)
        # save metadata info
        config_path = os.path.join(exp_dir, CONFIG_NAME)
        shutil.copyfile(WANDB_CONFIG, config_path)
        return exp_dir

    def _parse_config(self, config: Dict):
        self.max_epoch = config['epochs']
        self.model_name = config['model']
        self.dataset_path = config['dataset']
        self.batch_size = config['batch_size']
        self.lr = config['lr']
        self.min_lr = config['min_lr']
        self.loss_name = config['loss_fn']
        self.validate_every = config['validate_every']
        self.save_dir = config['save_dir']
        self.dataset_file_extension = config['dataset_file_extension']

    def _get_optimizer(self, model: nn.Module) -> Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.lr)

    def _get_scheduler(self, optimizer: Optimizer) -> object:
        min_lr, total_epochs = self.min_lr, self.max_epoch
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
        weights_dir = os.path.join(self.train_exp_dir, WEIGHTS_DIR_NAME)
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
        trainer = Trainer(config, gpus, checkpoint_path)
        trainer.train()
    except Exception as e:
        logging.error(f'Falied in training:{str(e)}, traceback:{traceback.format_exc()}')


def config_logging():
    file_handler = logging.FileHandler(filename='train.log')
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                        handlers=handlers)


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
