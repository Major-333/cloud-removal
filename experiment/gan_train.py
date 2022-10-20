import os
import wandb
import shutil
import logging
from tqdm import tqdm
from typing import Dict, Optional
import torch
from torch.optim import Optimizer
from warmup_scheduler import GradualWarmupScheduler
from torch import nn, distributed, optim
from datetime import datetime, timedelta
from sen12ms_cr_dataset.build import build_distributed_loaders_with_rois
from models.build import build_distributed_gan_model, build_pretrained_model_with_ddp
from loss.build import build_loss_fn
from evaluate import EvaluateType, Evaluater
from utils import get_rois_from_split_file, increment_path, setup_seed

START_TIME = (datetime.utcnow() + timedelta(hours=8)).strftime('%Y%m%d%H%M')
CHECKPOINT_NAME_PREFIX = 'Epoch'
TRAIN_SUBDIR_NAME = 'train'
VAL_SUBDIR_NAME = 'val'
EXP_SUBDIR_NAME = 'exp'
WEIGHTS_DIR_NAME = 'weights'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
CONFIG_FILENAME = 'config-defaults.yaml'
DEFAULT_SPLIT_FILENAME = 'split.yaml'


class GANTrainer(object):

    def __init__(self, config: Dict, local_rank: int, checkpoint_path: Optional[str] = None) -> None:
        # Load config to trainer
        self._parse_config(config)
        self.config = config
        # Fix random seed for reproducibility
        setup_seed(self.seed)
        # initialize PyTorch distributed using environment variables (you could also do this more explicitly by specifying `rank` and `world_size`,
        #  but I find using environment variables makes it so that you can easily use the same script on different machines)
        distributed.init_process_group(backend='nccl', init_method='env://')
        self.local_rank = local_rank
        torch.cuda.set_device(f'cuda:{local_rank}')
        # Init dataloader
        train_rois, val_rois, _ = get_rois_from_split_file(self.split_file_path)
        self.train_loader = build_distributed_loaders_with_rois(self.dataset_path, self.batch_size,
                                                                self.dataset_file_extension, train_rois)
        self.val_loader = build_distributed_loaders_with_rois(self.dataset_path, self.batch_size,
                                                              self.dataset_file_extension, val_rois)
        # Init model and optim
        self.model_S = build_pretrained_model_with_ddp(self.model_name_S, self.checkpoint_path_S, gpu_id=local_rank)
        self.model_G = build_distributed_gan_model(self.model_name_G, gpu_id=local_rank)
        self.loss_fn_G = build_loss_fn(self.loss_name_G)
        self.optimizer_G = self._get_optimizer(self.model_G)
        self.scheduler_G = self._get_scheduler(self.optimizer_G)

        self.model_D = build_distributed_gan_model(self.model_name_D, gpu_id=local_rank)
        self.loss_fn_D = build_loss_fn(self.loss_name_D)
        self.optimizer_D = self._get_optimizer(self.model_D)
        self.scheduler_D = self._get_scheduler(self.optimizer_D)

        # for model resume
        self.checkpoint_path = checkpoint_path
        if self.is_resume:
            self.resume_epoch_num = int(self.checkpoint_path.split(CHECKPOINT_NAME_PREFIX)[1])
        # for summary
        self.best_val_psnr = 0
        self.best_val_ssim = 0
        # for save
        if self.local_rank == 0:
            self.train_exp_dir = self._get_train_exp_dir()

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

    def _get_train_exp_dir(self) -> str:
        exp_dir = os.path.join(self.save_dir, TRAIN_SUBDIR_NAME, EXP_SUBDIR_NAME)
        exp_dir = increment_path(exp_dir)
        # save metadata info
        config_path = os.path.join(exp_dir, CONFIG_FILENAME)
        shutil.copyfile(CONFIG_FILENAME, config_path)
        split_file_path = os.path.join(exp_dir, DEFAULT_SPLIT_FILENAME)
        shutil.copyfile(self.split_file_path, split_file_path)
        return exp_dir

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

    def _parse_config(self, config: Dict):
        self.max_epoch = config['epochs']
        self.model_name_S = config['model_S']
        self.checkpoint_path_S = config['checkpoint_path_S']
        self.model_name_G = config['model_G']
        self.model_name_D = config['model_D']
        self.dataset_path = config['dataset']
        self.batch_size = config['batch_size']
        self.lr = config['lr']
        self.min_lr = config['min_lr']
        self.loss_name_G = config['loss_fn_G']
        self.loss_name_D = config['loss_fn_D']
        self.validate_every = config['validate_every']
        self.save_dir = config['save_dir']
        self.dataset_file_extension = config['dataset_file_extension']
        self.seed = config['seed']
        self.split_file_path = config['split_file_path']

    def train(self) -> None:
        for epoch in range(1, self.max_epoch + 1):
            self.model_G.train()
            self.model_D.train()
            epoch_loss_G = 0.0
            epoch_loss_L1 = 0.0
            epoch_loss_GAN = 0.0
            epoch_loss_D = 0.0
            training_info = {
                'learning rate G': self.optimizer_G.param_groups[0]['lr'],
                'learning rate D': self.optimizer_D.param_groups[0]['lr']
            }
            if not self.is_resume or epoch > self.resume_epoch_num:
                # let all processes sync up before starting with a new epoch of training
                distributed.barrier()
                for index, data_batch in enumerate(tqdm(self.train_loader, desc='Epoch: {}'.format(epoch))):
                    cloudy, ground_truth = data_batch
                    cloudy, ground_truth = cloudy.cuda(), ground_truth.cuda()
                    sar_image = cloudy[:, :2, :, :]
                    simulated_image = self.model_S(cloudy)
                    fused_image = self.model_G(simulated_image.detach(), cloudy)
                    self.set_requires_grad(self.model_D, True)
                    self.optimizer_D.zero_grad()
                    # backward_D
                    fake_concated_optical_sar_image = torch.cat((fused_image, sar_image), dim=1)
                    pred_fake = self.model_D(fake_concated_optical_sar_image.detach())
                    loss_D_fake = self.loss_fn_D(pred_fake, torch.zeros(pred_fake.shape).cuda())
                    real_concated_optical_sar_image = torch.cat((ground_truth, sar_image), dim=1)
                    pred_real = self.model_D(real_concated_optical_sar_image)
                    loss_D_real = self.loss_fn_D(pred_real, torch.ones(pred_real.shape).cuda())
                    loss_D = (loss_D_fake + loss_D_real) / 2
                    loss_D.backward()
                    self.optimizer_D.step()
                    epoch_loss_D += loss_D
                    self.set_requires_grad(self.model_D, False)
                    self.optimizer_G.zero_grad()
                    # backward_G
                    fake_concated_optical_sar_image = torch.cat((fused_image, sar_image), dim=1)
                    pred_fake = self.model_D(fake_concated_optical_sar_image)
                    loss_G, loss_L1, loss_GAN = self.loss_fn_G(pred_fake, fused_image, ground_truth)
                    loss_G.backward()
                    self.optimizer_G.step()
                    epoch_loss_G += loss_G
                    epoch_loss_L1 += loss_L1
                    epoch_loss_GAN += loss_GAN
                training_info = {
                    **training_info,
                    **{
                        'epoch_loss_G': epoch_loss_G.item() / len(self.train_loader),
                        'epoch_loss_L1': epoch_loss_L1.item() / len(self.train_loader),
                        'epoch_loss_GAN': epoch_loss_GAN.item() / len(self.train_loader),
                        'epoch_loss_D': epoch_loss_D.item() / len(self.train_loader)
                    }
                }
                if epoch % self.validate_every == 0:
                    distributed.barrier()
                    metric = Evaluater.evaluate_simulation_fusion_gan(self.model_S, self.model_G, self.val_loader,
                                                                      EvaluateType.VALIDATE)
                    training_info = {**training_info, **metric}
                    is_update = self._update_summary(metric, epoch, metric_prefix=EvaluateType.VALIDATE.value)
                    if is_update and self.local_rank == 0:
                        self._save(self.model_G, self.model_D, epoch)
            self._logs(training_info)
            self.scheduler_G.step()
            self.scheduler_D.step()
        self._finish()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

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

    def _save(self, model_G: nn.Module, model_D: nn.Module, epoch: int) -> None:
        weights_dir = os.path.join(self.train_exp_dir, WEIGHTS_DIR_NAME)
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        checkpoint_name = f'{CHECKPOINT_NAME_PREFIX}{epoch}.pt'
        file_path = os.path.join(weights_dir, checkpoint_name)
        logging.info(f'Will save model into{file_path}')
        print(f'Will save model into{file_path}')
        torch.save({'model_G': model_G.module.state_dict(), 'model_D': model_D.module.state_dict()}, file_path)


if __name__ == '__main__':
    if not os.getenv('CUDA_VISIBLE_DEVICES'):
        raise ValueError(f'set the env: `CUDA_VISIBLE_DEVICES` first')
    if not os.getenv('LOCAL_RANK'):
        raise ValueError(f'set the env: `LOCAL_RANK` first')
    if not os.getenv('WANDB_GROUP'):
        raise ValueError(f'set the env: `WANDB_GROUP` first')
    rank = int(os.getenv('LOCAL_RANK'))
    group = os.getenv('WANDB_GROUP')
    # Initialize wandb
    wandb.init(project='cloud removal V2', group=group, job_type='DDP mode')
    config = wandb.config
    # Initialize run
    trainer = GANTrainer(wandb.config, rank)
    trainer.train()