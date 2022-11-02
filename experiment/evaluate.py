import logging
import argparse
import enum
from typing import Dict
from tqdm import tqdm
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from metrics.pixel_metric import get_psnr, get_rmse, get_mae
from metrics.structure_metric import get_sam
from piq import ssim

CONFIG_FILEPATH = './config-defaults.yaml'

class EvaluateType(enum.Enum):
    VALIDATE = 'validate'
    TEST = 'test'


class Evaluater(object):
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def evaluate(model: nn.Module, dataloader: DataLoader, eval_type: EvaluateType) -> Dict:
        model.eval()
        batch_count = len(dataloader)
        total_rmse, total_psnr, total_ssim, total_sam, total_mae = 0, 0, 0, 0, 0
        logging.info(f'batch size:{dataloader.batch_size}, loader length is: {batch_count}')
        for _, data_batch in enumerate(tqdm(dataloader, desc=f'pid: {os.getpid()}. {eval_type} round')):
            cloudy, ground_truth = data_batch
            cloudy, ground_truth = cloudy.cuda(), ground_truth.cuda()
            with torch.no_grad():
                output = model(cloudy)
                output[output < 0] = 0
                output[output > 255] = 255
                total_rmse += get_rmse(output, ground_truth).item()
                total_psnr += get_psnr(output, ground_truth).item()
                total_ssim += ssim(output, ground_truth, data_range=255.).item()
                total_sam += get_sam(ground_truth, output).item()
                total_mae += get_mae(ground_truth, output).item()
        return {
            f'{eval_type.value}_rmse': total_rmse / batch_count,
            f'{eval_type.value}_psnr': total_psnr / batch_count,
            f'{eval_type.value}_ssim': total_ssim / batch_count,
            f'{eval_type.value}_sam': total_sam / batch_count,
            f'{eval_type.value}_mae': total_mae / batch_count,
        }
