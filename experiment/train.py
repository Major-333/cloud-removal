import logging
from typing import Optional, Dict, Tuple
from tqdm import tqdm
import os
import traceback
from datetime import datetime, timedelta
import wandb
import numpy as np
import torch
from torch import multiprocessing as mp
from metrics.pixel_metric import get_psnr, get_rmse
from utils import setup_ddp_envs, save_media, save_checkpoints
from init_helper import init_wandb, DDPInitHelper
from evaluate import Evaluater

START_TIME = (datetime.utcnow() + timedelta(hours=8)).strftime('%Y%m%d%H%M')

class Trainer(object):
    def __init__(self, rank: int, world_size: int, group: str, config: Dict) -> None:
        self.config = config
        self.init_helper = DDPInitHelper(self.config, rank, world_size)

    def _setup(self) -> None:
        self.train_loader, self.val_loader = self.init_helper.init_train_dsen2cr_dataloaders()
        # HACK: check Adam with lr_scheduler
        self.loss_fn = self.init_helper.get_loss_fn()
        self.model = self.init_helper.init_model()
        self.optimizer = self.init_helper.init_optimizer(self.model)
        self.scheduler = self.init_helper.get_warmup_scheduler(self.optimizer)
        self.loss_fn = self.init_helper.get_loss_fn()

    def _get_loss(self, output, ground_truth):
        if self.config['model'] != 'MPRNet':
            loss = self.loss_fn(output, ground_truth)
            return loss
        loss1 = self.loss_fn(output[0], ground_truth)
        loss2 = self.loss_fn(output[1], ground_truth)
        loss3 = self.loss_fn(output[2], ground_truth)
        loss = loss1 + loss2 + loss3
        return loss

    def _get_media_log(self, epoch, index, output, cloudy, ground_truth, patch_info) -> Dict:
        if self.config['model'] != 'MPRNet':
            media = save_media(epoch, index, output, cloudy, ground_truth, rank=rank, group=group, patch_info=patch_info)['image']
            return {'media': media}
        stage1_media = save_media(epoch, index, output[0], cloudy, ground_truth, rank=rank, group=group, patch_info=patch_info)['image']
        stage2_media = save_media(epoch, index, output[1], cloudy, ground_truth, rank=rank, group=group, patch_info=patch_info)['image']
        stage3_media = save_media(epoch, index, output[2], cloudy, ground_truth, rank=rank, group=group, patch_info=patch_info)['image']
        return {
            'stage1_media': stage1_media,
            'stage2_media': stage2_media,
            'stage3_media': stage3_media
        }

    def _get_metric_log(self, output, ground_truth) -> Dict:
        if self.config['model'] != 'MPRNet':
            return {
                'rmse': get_rmse(output, ground_truth),
                'psnr': get_psnr(output, ground_truth),
            }
        return {
            'rmse-1': get_rmse(output[0], ground_truth),
            'rmse-2': get_rmse(output[1], ground_truth),
            'rmse-3': get_rmse(output[2], ground_truth),
            'psnr-1': get_psnr(output[0], ground_truth),
            'psnr-2': get_psnr(output[1], ground_truth),
            'psnr-3': get_psnr(output[2], ground_truth)
        }

    def train(self) -> None:
        try:
            self._setup()
            for epoch in range(1, self.config['epochs']):
                epoch_loss = 0
                self.model.train()
                for index, data_batch in enumerate(tqdm(self.train_loader, desc='Epoch: {}'.format(epoch))):
                    self.optimizer.zero_grad()
                    logging.info(f'{len(data_batch)}')
                    cloudy, ground_truth, patch_info = data_batch
                    cloudy, ground_truth = cloudy.cuda().float(), ground_truth.cuda().float()
                    output = self.model(cloudy)
                    loss = self._get_loss(output, ground_truth)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    logs = {'loss': loss}
                    logs = logs | self._get_metric_log(output, ground_truth)
                    if index % wandb.config['visual_freq'] == 0:
                        media_logs = self._get_media_log(epoch, index, output, cloudy, ground_truth, patch_info)
                        logs = logs | media_logs
                    wandb.log(logs)
                if index % wandb.config['validate_freq'] == 0 and rank == 0:
                    metric = Evaluater.evaluate(self.model, self.val_loader, prefix='val')
                    logs = {'learning rate': self.optimizer.param_groups[0]['lr']}
                    logs = logs | metric
                    wandb.log(logs)
                save_checkpoints(self.model, epoch, rank, start_time=START_TIME)
                self.scheduler.step()   
        except Exception as e:
            logging.error(f'Falied in training:{str(e)}, traceback:{traceback.format_exc()}')
            wandb.finish()

def run(rank: int, world_size: int, group: str):
    try:
        config = init_wandb(rank, group)
        trainer = Trainer(rank, world_size, group, config)
        trainer.train()
    except Exception as e:
        logging.error(f'Falied in training:{str(e)}, traceback:{traceback.format_exc()}')


if __name__ == '__main__':
    logging.basicConfig(filename='logger.log',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    group = f'experiment-{wandb.util.generate_id()}'
    setup_ddp_envs()
    if not os.getenv('CUDA_VISIBLE_DEVICES'):
        raise ValueError(f'set the env: `CUDA_VISIBLE_DEVICES` first')
    parallel_num = len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))
    print(f'parallel_num:{parallel_num}')
    processes = []
    for rank in range(parallel_num):
        p = mp.Process(target=run, args=(rank, parallel_num, group))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()