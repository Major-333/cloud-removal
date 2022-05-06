from typing import Optional, Dict, Tuple
import logging
import os
import yaml
import wandb
import torch
from datetime import datetime, timedelta
from torch import nn, tensor, optim, distributed as dist, multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import optimizer, lr_scheduler
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from dataset.dsen2cr_dataset import Dsen2crDataset
from dataset.basic_dataloader import Seasons
from models.dsen2cr import DSen2_CR
from models.test_model import TestModel
from models.mprnet import MPRNet
from models.restormer import Restormer
from warmup_scheduler import GradualWarmupScheduler
from dataset.visualize import visualize_output_with_groundtruth, visualize_output_with_groundtruth_only_rgb, get_output_with_groundtruth_distribution_by_channel
from matplotlib import pyplot as plt
from dataset.basic_dataloader import SEN12MSCPatchRPath


LOSS_MAPPER = {'MSE': torch.nn.MSELoss()}

def setup_ddp_envs():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

def init_weights(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)

def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str) -> nn.Module:
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    return model

def load_ddp_checkpoint(model: nn.Module, checkpoint_path: str, device: str) -> nn.Module:
    ddp_state = torch.load(checkpoint_path, map_location=device)
    state = {}
    for key in ddp_state:
        new_key = key[7:] # ignore prefix: module.*
        state[new_key] = ddp_state[key]
    print(f'state:\n{state.keys()}')
    model.load_state_dict(state)
    
    return model

def parse_wandb_yaml(yaml_path: str) -> Dict:
    with open(yaml_path, 'r') as fp:
        raw_config = yaml.safe_load(fp)
    config = {}
    for key in raw_config:
        config[key] = raw_config[key]['value']
    return config

def get_warmup_scheduler(optimizer: optimizer, config: Dict) -> object:
    lr, total_epochs = config['lr'], config['epochs']
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs-warmup_epochs, eta_min=lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    return scheduler

def _get_media_path(name: str, suffix: str = None) -> str:
    media_dir = wandb.config['media_dir']
    if suffix:
        filename = f'{name}_{suffix}.png'
    else:
        filename = f'{name}.png'
    return os.path.join(media_dir, filename)


def _get_media_title(name: str, scene_id: str, patch_id: str) -> str:
    return f'scene_id:{scene_id}_patch_id:{patch_id}_{name}'


def save_all_media(epoch: int, iter_index: int, output: tensor, cloudy: tensor, ground_truth, rank: int, group: str) -> Dict:
    rank = f'rank:{rank}'
    epoch = str(epoch)
    media_logs = {}
    log_prefix = f'[pid:{os.getpid()}]'
    media_path = _get_media_path(epoch, group, rank, suffix=iter_index)
    media_title = _get_media_title(epoch, group, rank, suffix=iter_index)
    try:
        ground_truth = ground_truth.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        cloudy = cloudy.cpu().detach().numpy()
        for channel_idx in range(output.shape[1]):
            logging.info(f'{log_prefix} preparing channel:{channel_idx} distribution')
            savepath = _get_media_path(epoch, group, rank, suffix=f'{iter_index}_hist')
            get_output_with_groundtruth_distribution_by_channel(ground_truth, output, channel_idx,
                                                                savepath)
            media_logs[f'channel:{channel_idx} distribution'] = wandb.Image(plt.imread(savepath))
        visualize_output_with_groundtruth_only_rgb(ground_truth, output, cloudy, media_path,
                                                    media_title)
        media_logs['image'] = wandb.Image(plt.imread(media_path))
        return media_logs
    except Exception as e:
        logging.error(f'{log_prefix} failed to save the media: {str(e)}')
        return {}

def save_media(name: str, output: tensor, cloudy: tensor, ground_truth, patch_info: Dict) -> Dict:
    media_logs = {}
    media_path = _get_media_path(name)
    media_title = _get_media_title(name, patch_info['scene_id'][0], patch_info['patch_id'][0])
    try:
        ground_truth = ground_truth.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        cloudy = cloudy.cpu().detach().numpy()
        visualize_output_with_groundtruth_only_rgb(ground_truth, output, cloudy, media_path,
                                                    media_title)
        media_logs['image'] = wandb.Image(plt.imread(media_path))
        return media_logs
    except Exception as e:
        logging.error(f'failed to save the media: {str(e)}')
        return {}


def save_checkpoints(model :nn.Module, epoch: int, start_time: str, filename_prefix: Optional[str] = None, suffix: Optional[str] = None):
    model_name = wandb.config['model']
    checkpoints_dir = wandb.config['checkpoints_dir']
    subdir_name = f'{model_name}_{start_time}'
    filename = f'epoch{str(epoch)}'
    if filename_prefix:
        filename = f'{filename_prefix}_{filename}'
    if suffix:
        filename = f'{filename}_{suffix}'
    subdir_path = os.path.join(checkpoints_dir, subdir_name)
    if not os.path.isdir(subdir_path):
        os.makedirs(subdir_path)
    file_path = os.path.join(subdir_path, filename)
    logging.info(f'will save the model to:{file_path}')
    torch.save(model.state_dict(), file_path)

def init_dsen2cr() -> nn.Module:
    config = wandb.config
    model = DSen2_CR(in_channels=config['in_channels'],
                        out_channels=config['out_channels'],
                        num_layers=config['num_layers'],
                        feature_dim=config['feature_dim'])
    model.apply(init_weights)
    model = model.cuda()
    return model

def init_mprnet() -> nn.Module:
    model = MPRNet()
    model = model.cuda()
    return model

def init_test_model() -> nn.Module:
    model = TestModel()
    model.apply(init_weights)
    model = model.cuda()
    return model

def init_restormer() -> nn.Module:
    model = Restormer()
    model = model.cuda()
    return model