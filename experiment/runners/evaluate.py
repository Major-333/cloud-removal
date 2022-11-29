import logging
import enum
import argparse
import yaml
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm
import os
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from metrics.pixel_metric import get_psnr, get_rmse, get_mae
from metrics.structure_metric import get_sam
from piq import ssim
from runners.runner import Runner
from utils import get_rois_from_split_file, parse_wandb_yaml
from sen12ms_cr_dataset.build import build_loaders_with_rois
from sen12ms_cr_dataset.dataset import SEN12MSCRTriplet, Season
from models.build import build_model_with_dp

CONFIG_FILEPATH = './config-defaults.yaml'
EVAL_SUBDIR_NAME = 'val'
METRIC_FILENAME = 'metric.csv'
METADATA_FILENAME = 'metadata.yaml'
PREDICTS_DIRNAME = 'predicts'

class EvaluateType(enum.Enum):
    VALIDATE = 'validate'
    TEST = 'test'


class Evaluater(Runner):
    def __init__(self, config: Dict, gpus: List[int], save_predict: bool=False) -> None:
        super(Evaluater, self).__init__(config, EVAL_SUBDIR_NAME)
        self.save_predict = save_predict
        # Init dataloader
        _, val_rois, test_rois = get_rois_from_split_file(self.split_file_path)
        self.val_loader = build_loaders_with_rois(self.dataset_path, self.batch_size, self.dataset_file_extension, val_rois, debug=self.debug, return_with_triplet=save_predict)
        self.test_loader = build_loaders_with_rois(self.dataset_path, self.batch_size, self.dataset_file_extension, test_rois, debug=self.debug, return_with_triplet=save_predict)
        # Init model
        self.gpus = gpus
        self.model = build_model_with_dp(self.model_name, self.gpus)

    def _get_optimizer(self, model: nn.Module) -> Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.lr)
    
    def _load_model(self, checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state)

    def evaluate_checkpoint(self, checkpoint_path: str, eval_type: EvaluateType) -> Dict:
        logging.info(f'will evaluate the checkpoint:{checkpoint_path}')
        self._load_model(checkpoint_path)
        if eval_type == EvaluateType.VALIDATE:
            loader = self.val_loader
        elif eval_type == EvaluateType.TEST:
            loader = self.test_loader
        logging.info(f'save_predict is: {self.save_predict}')
        metric = self.evaluate(self.model, loader, eval_type=eval_type, save_predict=self.save_predict, predicts_dir_path=os.path.join(self.save_dir, PREDICTS_DIRNAME))
        self._save(metric, checkpoint_path, eval_type)

    def _save(self, metric: Dict, checkpoint_path: str, eval_type: EvaluateType) -> None:
        # save metadata
        metadata = {
            'checkpoint_path': checkpoint_path,
            'eval_type': eval_type.name,
        }
        metadata_filepath = os.path.join(self.save_dir, METADATA_FILENAME)
        logging.info(f'metadata:{metadata} is saved to {metadata_filepath}')
        with open(metadata_filepath, 'w') as wf:
            yaml.safe_dump(metadata, wf, default_flow_style=False)
        # save metric
        logging.info(f'metric:{metric}')
        metric_df = pd.DataFrame([metric], columns=metric.keys())
        metric_filepath = os.path.join(self.save_dir, METRIC_FILENAME)
        metric_df.to_csv(metric_filepath)
        logging.info(f'metric:\n{metric_df}')

    @property
    def is_master(self) -> bool:
        return self.local_rank == 0
    
    @staticmethod
    def save_ms_img(ms: np.array):
        pass

    @staticmethod
    def evaluate(model: nn.Module, dataloader: DataLoader, eval_type: EvaluateType, save_predict:bool=False, predicts_dir_path:str=None) -> Dict:
        model.eval()
        batch_count = len(dataloader)
        total_rmse, total_psnr, total_ssim, total_sam, total_mae = 0, 0, 0, 0, 0
        logging.info(f'batch size:{dataloader.batch_size}, loader length is: {batch_count}')
        for _, data_batch in enumerate(tqdm(dataloader, desc=f'pid: {os.getpid()}. {eval_type} round')):
            if save_predict:
                cloudy, ground_truth, triplets_dict = data_batch
                triplets = [
                    SEN12MSCRTriplet(triplets_dict['dataset_dir'][i], Season(triplets_dict['season'][i]), triplets_dict['scene_id'][i], triplets_dict['patch_id'][i], triplets_dict['file_extension'][i])
                    for i in range(len(triplets_dict['dataset_dir']))
                ]
            else:
                cloudy, ground_truth = data_batch
            cloudy, ground_truth = cloudy.cuda(), ground_truth.cuda()
            with torch.no_grad():
                output = model(cloudy)
                output[output < 0] = 0
                output[output > 1] = 1
                total_rmse += get_rmse(output, ground_truth).item()
                total_psnr += get_psnr(output, ground_truth).item()
                total_ssim += ssim(output, ground_truth, data_range=1.0).item()
                total_sam += get_sam(ground_truth, output).item()
                total_mae += get_mae(ground_truth, output).item()
            if save_predict:
                if not predicts_dir_path:
                    raise ValueError('predicts_dir_path is None')
                for triplet in triplets:
                    triplet.save_predict(predicts_dir_path, output.detach().cpu().numpy())
        return {
            f'{eval_type.value}_rmse': total_rmse / batch_count,
            f'{eval_type.value}_psnr': total_psnr / batch_count,
            f'{eval_type.value}_ssim': total_ssim / batch_count,
            f'{eval_type.value}_sam': total_sam / batch_count,
            f'{eval_type.value}_mae': total_mae / batch_count,
        }

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the Cloud Removal network')
    parser.add_argument('--checkpoint', type=str, required=True, help='Load model from a .pth file')
    parser.add_argument('--is_val', action='store_true', help='use val dataset instead of test dataset')
    return parser.parse_args()


if __name__ == '__main__':
    if not os.getenv('CUDA_VISIBLE_DEVICES'):
        raise ValueError(f'set the env: `CUDA_VISIBLE_DEVICES` first')
    gpus = [idx for idx in range(len(os.getenv('CUDA_VISIBLE_DEVICES').split(',')))]
    args = get_args()
    checkpoint_path = args.checkpoint
    config = parse_wandb_yaml(CONFIG_FILEPATH)
    evaluater = Evaluater(config, gpus)
    eval_type = EvaluateType.VALIDATE if args.is_val else EvaluateType.TEST
    evaluater.evaluate_checkpoint(checkpoint_path, eval_type)
