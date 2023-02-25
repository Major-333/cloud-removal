import os
import wandb
import shutil
import logging
from tqdm import tqdm
from typing import Dict, Optional
import torch
from torch.optim import Optimizer
from torch import nn, distributed
from models.build import build_distributed_gan_model, build_distributed_pretrained_model
from loss.build import build_loss_fn, build_distributed_gan_loss_fn
from runners.train import CHECKPOINT_NAME_PREFIX, WEIGHTS_DIR_NAME
from runners.evaluate import EvaluateType, Evaluater
from runners.distributed_train import DistributedTrainer, DistributedTrainStarter

class DistributedGANTrainer(DistributedTrainer):
    def __init__(self, config: Dict, local_rank: int, checkpoint_path: Optional[str] = None) -> None:
        super().__init__(config, local_rank, checkpoint_path)
    
    def _parse_model_specific_config(self, config: Dict):
        self.model_name_S = config['model_S']
        self.checkpoint_path_S = config['checkpoint_path_S']
        self.model_name_G = config['model_G']
        self.model_name_D = config['model_D']
        self.lr_G = config['lr_G']
        self.lr_D = config['lr_D']
        self.loss_name_G = config['loss_fn_G']
        self.loss_name_D = config['loss_fn_D']

    def _init_model(self):
        self.model_S = build_distributed_pretrained_model(self.model_name_S, self.checkpoint_path_S, gpu_id=self.local_rank)
        self.model_S.eval()

        self.model_G = build_distributed_gan_model(self.model_name_G, gpu_id=self.local_rank)
        self.loss_fn_G = build_distributed_gan_loss_fn(self.loss_name_G, gpu_id=self.local_rank)

        self.model_D = build_distributed_gan_model(self.model_name_D, gpu_id=self.local_rank)
        self.loss_fn_D = build_loss_fn(self.loss_name_D)

    def _init_optim(self):
        self.optimizer_G = self._get_optimizer(self.model_G, self.lr_G)
        self.scheduler_G = self._get_scheduler(self.optimizer_G)

        self.optimizer_D = self._get_optimizer(self.model_D, self.lr_D)
        self.scheduler_D = self._get_scheduler(self.optimizer_D)

    def _get_scheduler(self, optimizer: Optimizer) -> object:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return scheduler
    
    def train(self) -> None:
        for epoch in range(1, self.max_epoch + 1):
            self.model_G.train()
            self.model_D.train()
            epoch_loss_G = 0.0
            epoch_loss_local = 0.0
            epoch_loss_L1 = 0.0
            epoch_loss_GAN = 0.0
            epoch_loss_perceptual = 0.0
            epoch_loss_D = 0.0
            training_info = {
                'learning rate G': self.optimizer_G.param_groups[0]['lr'],
                'learning rate D': self.optimizer_D.param_groups[0]['lr']
            }
            if not self.is_resume or epoch > self.resume_epoch_num:
                # let all processes sync up before starting with a new epoch of training
                distributed.barrier()
                for index, data_batch in enumerate(tqdm(self.train_loader, desc='Epoch: {}'.format(epoch))):
                    cloudy, ground_truth, cloud_mask = data_batch
                    cloudy, ground_truth, cloud_mask = cloudy.cuda(), ground_truth.cuda(), cloud_mask.cuda()

                    simulated_image = self.model_S(cloudy)
                    sar_image = cloudy[:, :2, :, :]
                    fused_image = self.model_G(simulated_image.detach(), cloudy)

                    # backward_D
                    self.set_requires_grad(self.model_D, True)
                    self.optimizer_D.zero_grad()

                    fake_concated_optical_sar_image = torch.cat((fused_image, sar_image), dim=1)
                    pred_fake = self.model_D(fake_concated_optical_sar_image.detach())
                    loss_D_fake = self.loss_fn_D(pred_fake, torch.zeros(pred_fake.shape).cuda())

                    real_concated_optical_sar_image = torch.cat((ground_truth, sar_image), dim=1)
                    pred_real = self.model_D(real_concated_optical_sar_image)
                    loss_D_real = self.loss_fn_D(pred_real, torch.ones(pred_real.shape).cuda())

                    loss_D = (loss_D_fake + loss_D_real) / 2
                    loss_D.backward()
                    self.optimizer_D.step()
                    epoch_loss_D += loss_D.item()

                    # backward_G
                    self.set_requires_grad(self.model_D, False)
                    self.optimizer_G.zero_grad()

                    fake_concated_optical_sar_image = torch.cat((fused_image, sar_image), dim=1)
                    pred_fake = self.model_D(fake_concated_optical_sar_image)
                    loss_G, loss_L1, loss_local, loss_GAN, loss_perceptual = self.loss_fn_G(
                        pred_fake, fused_image, ground_truth, cloud_mask)

                    loss_G.backward()
                    self.optimizer_G.step()
                    epoch_loss_G += loss_G.item()
                    epoch_loss_local += loss_local.item()
                    epoch_loss_L1 += loss_L1.item()
                    epoch_loss_GAN += loss_GAN.item()
                    epoch_loss_perceptual += loss_perceptual.item()

                training_info = {
                    **training_info,
                    **{
                        'epoch_loss_G': epoch_loss_G / len(self.train_loader),
                        'epoch_loss_local': epoch_loss_local / len(self.train_loader),
                        'epoch_loss_L1': epoch_loss_L1 / len(self.train_loader),
                        'epoch_loss_GAN': epoch_loss_GAN / len(self.train_loader),
                        'epoch_loss_perceptual': epoch_loss_perceptual / len(self.train_loader),
                        'epoch_loss_D': epoch_loss_D / len(self.train_loader)
                    }
                }

                if epoch % self.validate_every == 0:
                    distributed.barrier()
                    metric = Evaluater.evaluate(lambda x: self.model_G(self.model_S(x), x), self.val_loader,
                                                                      EvaluateType.VALIDATE)
                    training_info = {**training_info, **metric}
                    is_update = self._update_summary(metric, epoch, metric_prefix=EvaluateType.VALIDATE.value)
                    if is_update and self.local_rank == 0:
                        self._save(self.model_G, self.model_D, epoch)
            self._logs(training_info)
            if epoch >= 75:
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

    def _save(self, model_G: nn.Module, model_D: nn.Module, epoch: int) -> None:
        weights_dir = os.path.join(self.save_dir, WEIGHTS_DIR_NAME)
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        checkpoint_name = f'{CHECKPOINT_NAME_PREFIX}{epoch}.pt'
        file_path = os.path.join(weights_dir, checkpoint_name)
        logging.info(f'Will save model into{file_path}')
        print(f'Will save model into{file_path}')
        torch.save({'model_G': model_G.module.state_dict(), 'model_D': model_D.module.state_dict()}, file_path)

class DistributedGANTrainStarter(DistributedTrainStarter):
    def __init__(self) -> None:
        super().__init__()
    
    def init_trainner(self, rank):
        self.trainer = DistributedGANTrainer(wandb.config, rank)

if __name__ == '__main__':
    train_starter = DistributedGANTrainStarter()
    train_starter.start_train()