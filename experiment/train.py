from email.errors import CloseBoundaryNotFoundDefect
import logging
from sre_constants import JUMP
from tkinter import Image
from tqdm import tqdm
import os
import sys
import wandb
import torch
from torch import nn
from torch import distributed as dist, logspace
from torch import multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset.dsen2cr_dataset import Dsen2crDataset
from dataset.basic_dataloader import Seasons
from models.dsen2cr import DSen2_CR
from metrics.psnr import get_psnr, get_rmse
from dataset.visualize import visualize_output_with_groundtruth, visualize_output_with_groundtruth_only_rgb, get_output_with_groundtruth_distribution_by_channel
from matplotlib import pyplot as plt

LOSS_MAPPER = {'MSE': torch.nn.MSELoss()}


def _ddp_setup():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'


def _init_dataloader() -> DataLoader:
    config = wandb.config
    scene_black_list = config['scene_black_list']
    scene_white_list = config['scene_white_list']
    train_dataset = Dsen2crDataset(config['data_dir'],
                                   config['processed_dir'],
                                   Seasons(config['season']),
                                   scene_white_list=scene_white_list,
                                   scene_black_list=scene_black_list)
    train_sampler = DistributedSampler(train_dataset)
    dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=8, sampler=train_sampler)
    return dataloader


def _get_media_path(epoch: int, group: str, job_type: str, suffix: str = None) -> str:
    media_dir = wandb.config['media_dir']
    return os.path.join(media_dir, f'{group}_{job_type}_{epoch}epochs_{suffix}.png')


def _get_media_title(epoch: int, group: str, job_type: str, suffix: str = None) -> str:
    return f'{group}_{job_type}_{epoch}epochs_{suffix}'


def _init_weights(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)


def train(rank: int, world_size: int, group: str):
    try:
        job_type = f'rank:{rank}'
        wandb.init(project='cloud removal', group=group, job_type=job_type)
        config = wandb.config
        logging.info(f'config is:{config}')
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        model = DSen2_CR(in_channels=config['in_channels'],
                         out_channels=config['out_channels'],
                         num_layers=config['num_layers'],
                         feature_dim=config['feature_dim'])
        model.apply(_init_weights)
        model = model.cuda()
        model = DDP(model, device_ids=[rank])
        wandb.watch(model, log_freq=100)
        train_loader = _init_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        for epoch in range(1, config['epochs']):
            epoch_loss = 0
            model.train()
            for index, data_batch in enumerate(tqdm(train_loader, desc='Epoch: {}'.format(epoch))):
                optimizer.zero_grad()
                cloudy, ground_truth = data_batch
                logging.info(f'{cloudy.shape}, {ground_truth.shape}')
                cloudy, ground_truth = cloudy.cuda().float(), ground_truth.cuda().float()
                logging.info(f'[cloudy]:{torch.sum(torch.isnan(cloudy.view(-1)))}')
                output = model(cloudy)
                loss_fn = LOSS_MAPPER[config['loss_fn']]
                loss = loss_fn(output, ground_truth)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                rmse = get_rmse(output, ground_truth)
                psnr = get_psnr(output, ground_truth)
                logs = {'loss': loss, 'rmse': rmse, 'psnr': psnr}
                if index % wandb.config['visual_freq'] == 0:
                    media_path = _get_media_path(epoch, group, job_type, suffix=index)
                    media_title = _get_media_title(epoch, group, job_type, suffix=index)
                    try:
                        ground_truth = ground_truth.cpu().detach().numpy()
                        output = output.cpu().detach().numpy()
                        cloudy = cloudy.cpu().detach().numpy()
                        for channel_idx in range(output.shape[1]):
                            logging.info(f'preparing channel:{channel_idx} distribution')
                            savepath = _get_media_path(epoch, group, job_type, suffix=f'{index}_hist')
                            get_output_with_groundtruth_distribution_by_channel(ground_truth, output, channel_idx,
                                                                                savepath)
                            logs[f'channel:{channel_idx} distribution'] = wandb.Image(plt.imread(savepath))
                        visualize_output_with_groundtruth_only_rgb(ground_truth, output, cloudy, media_path,
                                                                   media_title)
                        logs['image'] = wandb.Image(plt.imread(media_path))
                    except Exception as e:
                        logging.error(f'log failed with msg:{str(e)}')
                wandb.log(logs)
    except Exception as e:
        logging.error(f'Falied in training:{str(e)}')
    wandb.finish()


if __name__ == '__main__':
    logging.basicConfig(filename='logger.log',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    group = f'experiment-{wandb.util.generate_id()}'
    _ddp_setup()
    if not os.getenv('CUDA_VISIBLE_DEVICES'):
        raise ValueError(f'set the env: `CUDA_VISIBLE_DEVICES` first')
    parallel_num = len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))
    print(f'parallel_num:{parallel_num}')
    processes = []
    for rank in range(parallel_num):
        p = mp.Process(target=train, args=(rank, parallel_num, group))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
