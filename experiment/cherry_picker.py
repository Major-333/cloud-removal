import queue
import numpy as np
from collections import OrderedDict
from typing import Dict, Callable, List
import logging
import argparse
import json
from typing import Optional, Dict, Tuple
from requests import patch
from tqdm import tqdm
import os
import torch
from torch import nn, tensor, optim, distributed as dist, multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from metrics.pixel_metric import get_psnr, get_rmse, get_mae
from metrics.structure_metric import get_sam
from utils import parse_wandb_yaml, load_ddp_checkpoint, CHECKPOINT_NAME_PREFIX
from piq import ssim
from models.mprnet import MPRNet
from init_helper import InitHelper
from utils import visualize_output_with_groundtruth_only_rgb
import heapq
import re

CONFIG_FILEPATH = './config-defaults.yaml'
SUMMARY_FILENAME = 'summary.json'
SSIM_KEY = 'test_ssim'
PSNR_KEY = 'test_psnr'
EPOCH_NUM_KEY = 'epoch_num'
PATCH_METRICS_FILENAME = 'patch-metric.json'

class CherryPicker(object):
    def __init__(self, config: Dict, device: str) -> None:
        self._config = config
        self._device = device

    @staticmethod
    def selecter(summary: Dict, topk: int, key: str) -> OrderedDict:
        selected_summary = OrderedDict()
        topk_metrics = []
        for metric in summary:
            heapq.heappush(topk_metrics, (float(metric[key]), metric))
            if len(topk_metrics) > topk:
                heapq.heappop(topk_metrics)
        topk_metrics.reverse()
        for _, metric in topk_metrics:
            metric_copy = metric.copy()
            epoch_num = metric_copy.pop(EPOCH_NUM_KEY)
            selected_summary[epoch_num] = metric_copy
        return selected_summary

    def cherry_pick_topk_checkpoint_on_test(self, checkpoint_dirs: List[str], topk: int=5) -> None:
        for checkpoint_dir in checkpoint_dirs:
            metric_filenames = [filename for filename in os.listdir(checkpoint_dir) if filename.endswith('.json') and filename.startswith(CHECKPOINT_NAME_PREFIX)]
            summary_list = []
            for metric_filename in metric_filenames:
                epoch_num = int(metric_filename.split('.')[0].replace(CHECKPOINT_NAME_PREFIX, ''))
                metric_filepath = os.path.join(checkpoint_dir, metric_filename)
                new_metric = {EPOCH_NUM_KEY: epoch_num}
                with open(metric_filepath, 'r') as json_file:
                    metric = json.load(json_file)
                    for key in metric:
                        if key.startswith('test'):
                            new_metric[key] = metric[key]
                summary_list.append(new_metric)
            summary = {}
            summary['ssim'] = self.selecter(summary_list, topk, key=SSIM_KEY)
            summary['psnr'] = self.selecter(summary_list, topk, key=PSNR_KEY)
            summary_filepath = os.path.join(checkpoint_dir, SUMMARY_FILENAME)
            with open(summary_filepath, 'w') as json_file:
                json.dump(summary, json_file, indent=4)
 

    def cherry_pick_topk_visual_perception(self, checkpoint_paths: List[str], on_train: bool, k: int=60000):
        for index, checkpoint_path in enumerate(checkpoint_paths):
            # HACK: modify batch size and model name
            config = self._config.copy()
            config['model'] = os.path.basename(os.path.dirname(checkpoint_path))
            config['batch_size'] = 1
            init_helper = InitHelper(config)
            dataloader = init_helper.init_dsen2cr_dataloader(self._config['train_dir']) if on_train else init_helper.init_dsen2cr_dataloader(self._config['test_dir'])
            model = init_helper.init_model()
            model = load_ddp_checkpoint(model, checkpoint_path, self._device)
            model.eval()
            patch_metrics = []
            visual_perception_dir = os.path.join(config['visual_perception_dir'], config['model'])
            os.makedirs(visual_perception_dir, exist_ok=True)
            visual_perception_path = os.path.join(visual_perception_dir, PATCH_METRICS_FILENAME)
            if os.path.isfile(visual_perception_path):
                with open(visual_perception_path, 'r') as json_file:
                    patch_metrics = json.load(json_file)
            else:
                for index, data_batch in enumerate(tqdm(dataloader, desc='Test round')):
                    cloudy, ground_truth, patch_info = data_batch
                    cloudy, ground_truth = cloudy.cuda(), ground_truth.cuda()
                    with torch.no_grad():
                        if isinstance(model, MPRNet):
                            output = model(cloudy)[2]
                        else:
                            output = model(cloudy)
                        output[output < 0] = 0
                        output[output > 255] = 255
                        patch_metric = {
                            'patch_info': patch_info,
                            'predict_psnr': get_psnr(output, ground_truth).item(),
                            'predict_ssim': ssim(output, ground_truth, data_range=255.).item(),
                            'raw_psnr': get_psnr(cloudy[:, 2:, :, :], ground_truth).item(),
                            'raw_ssim': ssim(cloudy[:, 2:, :, :], ground_truth, data_range=255.).item(),
                        }
                        patch_metrics.append(patch_metric)
            for patch_metric in patch_metrics:
                patch_metric['diff_ssim'] = float(patch_metric['raw_ssim']) - float(patch_metric['predict_ssim'])
                patch_metric['diff_psnr'] = float(patch_metric['raw_ssim']) - float(patch_metric['predict_ssim'])
            patch_metrics = sorted(patch_metrics, key=lambda item: item['diff_ssim'])
            with open(visual_perception_path, 'w') as json_file:
                json.dump(patch_metrics, json_file, indent=4)
            if index == 0:
                bucket_dir_names = []
                bucket_topk_list = []
                for i in range(5):
                    bucket_up_bound = 0.2*(i+1)
                    bucket_low_bound = 0.2*i
                    bucket_metrics = [patch_metric for patch_metric in patch_metrics if float(patch_metric['raw_ssim']) < bucket_up_bound and float(patch_metric['raw_ssim']) >= bucket_low_bound]
                    bucket_metrics = sorted(bucket_metrics, key=lambda item: float(item['diff_ssim']))
                    bucket_topk_list.append([bucket_metric['patch_info'] for bucket_metric in bucket_metrics[:min(k, len(bucket_metrics))]])
                    bucket_dir_name = f'largest_diff_range_{bucket_low_bound}_{bucket_up_bound}'
                    bucket_dir_names.append(bucket_dir_name)
            for data_batch in tqdm(dataloader, desc='Test round'):
                cloudy, ground_truth, patch_info = data_batch
                for bucket_dir_name, bucket_topk in zip(bucket_dir_names, bucket_topk_list):    
                    if patch_info not in bucket_topk:
                        continue
                    cloudy, ground_truth = cloudy.cuda(), ground_truth.cuda()
                    with torch.no_grad():
                        if isinstance(model, MPRNet):
                            output = model(cloudy)[2]
                        else:
                            output = model(cloudy)
                        output[output < 0] = 0
                        output[output > 255] = 255
                        ground_truth = ground_truth.cpu().detach().numpy()
                        output = output.cpu().detach().numpy()
                        cloudy = cloudy.cpu().detach().numpy()
                        # HACK:
                        tmp = re.findall("\d+", f'{patch_info["scene_id"]}-{patch_info["patch_id"]}.png')
                        title = f'{tmp[0]}-{tmp[1]}.png'
                        dir_path = os.path.join(visual_perception_dir, bucket_dir_name)
                        os.makedirs(dir_path, exist_ok=True)
                        visual_path = os.path.join(dir_path, title)
                        visualize_output_with_groundtruth_only_rgb(ground_truth, output, cloudy, visual_path, title)
        


def get_args():
    parser = argparse.ArgumentParser(description='Cherry pick checkpoint and visual perception result')
    subparsers = parser.add_subparsers(dest='action')
    summary_parser = subparsers.add_parser('summary')
    summary_parser.add_argument('--dirs', nargs='+', required=True, help='checkpoint dirs for summary')
    visual_perception_parser = subparsers.add_parser('visual-perception')
    visual_perception_parser.add_argument('--checkpoints', nargs='+', required=True, help='checkpoint paths for picking visual-perception')
    visual_perception_parser.add_argument('--on-test', dest='on_train', default=False, action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s %(levelname)-8s %(message)s')
    args = get_args()
    if not os.getenv('CUDA_VISIBLE_DEVICES'):
        raise ValueError(f'set the env: `CUDA_VISIBLE_DEVICES` first')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'will use gpu:{device}')
    if not os.path.exists('config-defaults.yaml'):
        raise ValueError(f'can not find the config file: config-defaults.yaml')
    config = parse_wandb_yaml(CONFIG_FILEPATH)
    logging.info(f'config:{config}')
    cherry_picker = CherryPicker(config, device)
    if args.action == 'summary':
        checkpoint_relpaths = args.dirs
        checkpoint_dir_paths = [os.path.join(config['checkpoints_dir'], checkpoint_relpath) for checkpoint_relpath in checkpoint_relpaths]
        cherry_picker.cherry_pick_topk_checkpoint_on_test(checkpoint_dir_paths)
    if args.action == 'visual-perception':
        checkpoint_relpaths = args.checkpoints
        checkpoint_paths = [os.path.join(config['checkpoints_dir'], checkpoint_relpath) for checkpoint_relpath in checkpoint_relpaths]
        cherry_picker.cherry_pick_topk_visual_perception(checkpoint_paths, args.on_train)
