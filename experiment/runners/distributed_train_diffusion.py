import os
import wandb
import shutil
import logging
from tqdm import tqdm
from typing import Dict, Optional
import torch
from torch.optim import Optimizer
from torch import nn, distributed
from models.build import build_distributed_model
from loss.build import build_loss_fn
from runners.train import CHECKPOINT_NAME_PREFIX, WEIGHTS_DIR_NAME
from runners.evaluate import EvaluateType, Evaluater
from runners.distributed_train import DistributedTrainer, DistributedTrainStarter

class DistributedDiffusionTrainer(DistributedTrainer):
    def __init__(self, config: Dict, local_rank: int, checkpoint_path: Optional[str] = None) -> None:
        super().__init__(config, local_rank, checkpoint_path)

    def _parse_model_specific_config(self, config: Dict):
        self.model_name = config['model']
        self.lr = config['lr']
        self.loss_name = config['loss_fn']
        self.beta_schedule = {
            'train': {
                'schedule': config['train_schedule'],
                'n_timestep': config['train_n_timestep'],
                'linear_start': config['train_linear_start'],
                'linear_end': config['train_linear_end']
            },
            'test': {
                'schedule': config['test_schedule'],
                'n_timestep': config['test_n_timestep'],
                'linear_start': config['test_linear_start'],
                'linear_end': config['test_linear_end']
            },
        }

    def _init_model(self):
        self.model = build_distributed_model(self.model_name, gpu_id=self.local_rank)
        self.model.module.set_beta_schedule(self.beta_schedule)
        self.loss_fn = build_loss_fn(self.loss_name)

    def _init_optim(self):
        self.optimizer = self._get_optimizer(self.model, self.lr)

    def _get_optimizer(self, model: nn.Module, lr: float) -> Optimizer:
        return torch.optim.Adam(model.parameters(), lr=lr)

    def train(self) -> None:
        for epoch in range(1, self.max_epoch+1):
            self.model.train()
            self.model.module.set_new_noise_schedule(phase='train')
            epoch_loss = 0.0
            training_info = {'learning rate': self.optimizer.param_groups[0]['lr']}
            if not self.is_resume or epoch > self.resume_epoch_num:
                # let all processes sync up before starting with a new epoch of training
                distributed.barrier()
                for index, data_batch in enumerate(tqdm(self.train_loader, desc='Epoch: {}'.format(epoch))):
                    self.optimizer.zero_grad()
                    cloudy, ground_truth = data_batch
                    cloudy, ground_truth = cloudy.cuda(), ground_truth.cuda()
                    y_ms = cloudy[:, 2:, :, :]
                    y_sar = cloudy[:, :2, :, :]
                    noise_hat, noise = self.model(ground_truth, y_ms, y_sar)
                    loss = self.loss_fn(noise_hat, noise)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss
                training_info = {**training_info , **{'epoch_loss': epoch_loss.item()}}
                if epoch % self.validate_every == 0:
                    # let all processes sync up before starting with a new epoch of validating
                    distributed.barrier()
                    self.model.eval()
                    self.model.module.set_new_noise_schedule(phase='test')
                    metric = Evaluater.evaluate(lambda x: self.model.module.restoration(x[:, 2:, :, :], x[:, :2, :, :]), self.val_loader, EvaluateType.VALIDATE)
                    training_info = {**training_info, **metric}
                    logging.info(f'metric:{metric}')
                    is_update = self._update_summary(metric, epoch, metric_prefix=EvaluateType.VALIDATE.value)
                    if is_update and self.local_rank == 0:
                        self._save(self.model, epoch)
            self._logs(training_info)
        self._finish()

class DistributedDiffusionTrainStarter(DistributedTrainStarter):
    def __init__(self) -> None:
        super().__init__()
    
    def init_trainner(self, rank):
        self.trainer = DistributedDiffusionTrainer(wandb.config, rank)


if __name__ == '__main__':
    train_starter = DistributedDiffusionTrainStarter()
    train_starter.start_train()