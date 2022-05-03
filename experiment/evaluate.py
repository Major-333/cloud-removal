import logging
import argparse
import json
import yaml
from typing import Optional, Dict, Tuple
from tqdm import tqdm
import os
import torch
from torch import nn, tensor, optim, distributed as dist, multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from metrics.pixel_metric import get_psnr, get_rmse, get_mae
from metrics.structure_metric import get_sam
from utils import parse_wandb_yaml, load_ddp_checkpoint
from piq import ssim
from models.mprnet import MPRNet
from init_helper import InitHelper

CONFIG_FILEPATH = './config-defaults.yaml'

class Evaluater(object):

    def __init__(self, config: Dict, device: str) -> None:
        self._config = config
        self._device = device
        self._init_helper = InitHelper(self._config)

    def _setup(self):
        self.train_loader = self._init_helper.init_dsen2cr_dataloader(self.config['test_dir'])
        self.test_loader = self._init_helper.init_dsen2cr_dataloader(self.config['train_dir'])
        self.model = self.init_helper.init_model()

    def evaluate_checkpoint(self, checkpoint_path: str) -> Dict:
        logging.info(f'will evaluate the checkpoint:{checkpoint_path}')
        print(f'model state:\n{self.model.state_dict().keys()}')
        trained_model = load_ddp_checkpoint(self.model, checkpoint_path, self._device)
        train_metric = self.evaluate(trained_model, self.train_loader, prefix='train')
        test_metric = self.evaluate(trained_model, self.test_loader, prefix='test')
        metric = train_metric | test_metric
        checkpoints_dir = os.path.dirname(checkpoint_path)
        metric_filename = f'{os.path.basename(checkpoint_path).split(".")[0]}.json'
        metric_filepath = os.path.join(checkpoints_dir, metric_filename)
        logging.info(f'will save the metric to:{metric_filepath}')
        with open(metric_filepath, 'w') as fp:
            json.dump(metric, fp)

    @staticmethod
    def evaluate(model: nn.Module, dataloader: DataLoader, prefix: Optional[str] = None) -> Dict:
        model.eval()
        num_val_batches = len(dataloader)
        total_rmse, total_psnr, total_ssim, total_sam, total_mae = 0, 0, 0, 0, 0
        for index, data_batch in enumerate(tqdm(dataloader, desc=f'pid: {os.getpid()}. Validation round')):
            cloudy, ground_truth, _ = data_batch
            cloudy, ground_truth = cloudy.cuda().float(), ground_truth.cuda().float()
            with torch.no_grad():
                if isinstance(model, MPRNet):
                    output = model(cloudy)[2]
                else:
                    output = model(cloudy)
                output[output < 0] = 0
                output[output > 255] = 255
                output, ground_truth = output.long().float(), ground_truth.long().float()
                total_rmse += get_rmse(output, ground_truth).item()
                total_psnr += get_psnr(output, ground_truth).item()
                total_ssim += ssim(output, ground_truth, data_range=255.).item()
                total_sam += get_sam(ground_truth, output).item()
                total_mae += get_mae(ground_truth, output).item()
        metric = {
            f'{prefix}_rmse': total_rmse / num_val_batches,
            f'{prefix}_psnr': total_psnr / num_val_batches,
            f'{prefix}_ssim': total_ssim / num_val_batches,
            f'{prefix}_sam': total_sam / num_val_batches,
            f'{prefix}_mae': total_mae / num_val_batches,
        }
        return metric

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--load', '-f', type=str, required=True, help='Load model from a .pth file')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(filename='logger.log',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s %(levelname)-8s %(message)s')
    if not os.getenv('CUDA_VISIBLE_DEVICES'):
        raise ValueError(f'set the env: `CUDA_VISIBLE_DEVICES` first')
    if not os.path.exists('config-defaults.yaml'):
        raise ValueError(f'can not find the config file: config-defaults.yaml')
    config = parse_wandb_yaml(CONFIG_FILEPATH)
    print(f'config:\n{config}')
    args = get_args()
    checkpoint_path = args.load
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'will use gpu:{device}')
    evaluater = Evaluater(config, device)
    evaluater.evaluate_checkpoint(checkpoint_path)
