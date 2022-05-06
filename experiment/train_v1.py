import logging
from typing import Optional, Dict, Tuple
from tqdm import tqdm
import os
import sys
import torch
import traceback
from datetime import datetime, timedelta
import wandb
from metrics.pixel_metric import get_psnr, get_rmse
from utils import save_media, save_checkpoints
from init_helper import DPInitHelper, init_wandb_in_dp
from evaluate import Evaluater

START_TIME = (datetime.utcnow() + timedelta(hours=8)).strftime('%Y%m%d%H%M')

class DPTrainer(object):
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.init_helper = DPInitHelper(self.config)

    def _setup(self) -> None:
        self.train_loader, self.val_loader = self.init_helper.init_train_val_dsen2cr_dataloaders()
        # HACK: check Adam with lr_scheduler
        self.loss_fn = self.init_helper.get_loss_fn()
        self.model = self.init_helper.init_model()
        self.optimizer = self.init_helper.init_optimizer(self.model)
        self.scheduler = self.init_helper.get_warmup_scheduler(self.optimizer)
        self.loss_fn = self.init_helper.get_loss_fn()
        self.best_val_psnr = 0
        self.best_val_ssim = 0

    def _get_loss(self, output, ground_truth):
        if self.config['model'] != 'MPRNet':
            loss = self.loss_fn(output, ground_truth)
            return loss
        loss1 = self.loss_fn(output[0], ground_truth)
        loss2 = self.loss_fn(output[1], ground_truth)
        loss3 = self.loss_fn(output[2], ground_truth)
        loss = loss1 + loss2 + loss3
        return loss

    def _get_media_log(self, epoch, output, cloudy, ground_truth, patch_info, key_suffix: Optional[str] = None) -> Dict:
        name = f'epoch{epoch}'
        if self.config['model'] != 'MPRNet':
            media = save_media(name, output, cloudy, ground_truth, patch_info=patch_info)['image']
            return {f'media_{key_suffix}': media}
        stage1_media = save_media(name, output[0], cloudy, ground_truth, patch_info=patch_info)['image']
        stage2_media = save_media(name, output[1], cloudy, ground_truth, patch_info=patch_info)['image']
        stage3_media = save_media(name, output[2], cloudy, ground_truth, patch_info=patch_info)['image']
        return {
            f'stage1_media_{key_suffix}': stage1_media,
            f'stage2_media_{key_suffix}': stage2_media,
            f'stage3_media_{key_suffix}': stage3_media
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

    def _update_summary(self, metric: Dict, epoch: int):
        is_update = False
        if metric[f'val_psnr'] > self.best_val_psnr:
            logging.info(f'best val psnr update:{self.best_val_psnr}, on epoch:{epoch}')
            self.best_val_psnr = metric[f'val_psnr']
            wandb.run.summary['best_val_psnr'] = self.best_val_psnr
            wandb.run.summary['best_val_psnr_epoch'] = epoch
            is_update = True
        if metric[f'val_ssim'] > self.best_val_ssim:
            logging.info(f'best val ssim update:{self.best_val_ssim}, on epoch:{epoch}')
            self.best_val_ssim = metric[f'val_ssim']
            wandb.run.summary['best_val_ssim'] = self.best_val_ssim
            wandb.run.summary['best_val_ssim_epoch'] = epoch
            is_update = True
        if is_update:
            save_checkpoints(self.model, epoch, start_time=START_TIME)

    def train(self) -> None:
        try:
            self._setup()
            for epoch in range(1, self.config['epochs']):
                self.model.train()
                epoch_loss = 0
                logs = {}
                for index, data_batch in enumerate(tqdm(self.train_loader, desc='Epoch: {}'.format(epoch))):
                    self.optimizer.zero_grad()
                    cloudy, ground_truth, patch_info = data_batch
                    cloudy, ground_truth = cloudy.cuda(), ground_truth.cuda()
                    output = self.model(cloudy)
                    loss = self._get_loss(output, ground_truth)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss                
                    if index % wandb.config['visual_freq'] == 0:
                        media_logs = self._get_media_log(epoch, output, cloudy, ground_truth, patch_info, key_suffix=f'{index}')
                        logs = logs | media_logs
                logs = logs | {'epoch_loss': epoch_loss.item()}
                if index % wandb.config['validate_freq'] == 0:
                    metric = Evaluater.evaluate(self.model, self.val_loader, prefix='val')
                    logs = logs | {'learning rate': self.optimizer.param_groups[0]['lr']}
                    logs = logs | metric
                    self._update_summary(metric, epoch)
                wandb.log(logs)
                self.scheduler.step()
        except Exception as e:
            logging.error(f'Falied in training:{str(e)}, traceback:{traceback.format_exc()}')
            wandb.finish()

def run(group: str):
    try:
        config = init_wandb_in_dp(group)
        trainer = DPTrainer(config)
        trainer.train()
    except Exception as e:
        logging.error(f'Falied in training:{str(e)}, traceback:{traceback.format_exc()}')


def config_log():
    file_handler = logging.FileHandler(filename='tmp.log')
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

if __name__ == '__main__':
    config_log()
    if not os.getenv('CUDA_VISIBLE_DEVICES'):
        raise ValueError(f'set the env: `CUDA_VISIBLE_DEVICES` first')
    parallel_num = len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))
    logging.info(f'parallel_num:{parallel_num}')
    group = f'experiment-{wandb.util.generate_id()}'
    run(group)