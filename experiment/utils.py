from datetime import datetime
from typing import Dict, Tuple, List
import os
import sys
import yaml
import logging
import torch
from torch import nn
import numpy as np
import random
from sen12ms_cr_dataset.dataset import Roi, Season

DEFAULT_LOG_FILENAME = 'train.log'

def init_weights(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)


def parse_wandb_yaml(yaml_path: str) -> Dict:
    with open(yaml_path, 'r') as fp:
        raw_config = yaml.safe_load(fp)
    config = {}
    for key in raw_config:
        config[key] = raw_config[key]['value']
    return config

def increment_path(dir_path: str, sep='') -> str:
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for n in range(2, 9999):
        next_dir_path = f'{dir_path}{sep}{n}'
        if not os.path.exists(next_dir_path):
            os.makedirs(next_dir_path)
            return next_dir_path
    return None

def config_logging(filename:str=DEFAULT_LOG_FILENAME):
    file_handler = logging.FileHandler(filename=filename)
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                        handlers=handlers,
                        force=True)
    logging.info(f'start logging:{datetime.now()}')

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def roi_to_str(roi: Roi) -> str:
    return f'{roi.season.value}_{roi.scene_id}'

def str_to_roi(roi_str: str) -> Roi:
    scene_id = roi_str.rsplit('_', 1)[-1]
    season = Season(roi_str.rsplit('_', 1)[0])
    return Roi(season, scene_id)

def get_rois_from_split_file(split_file_path: str) -> Tuple[List[Roi], List[Roi], List[Roi]]:
    if not os.path.isfile(split_file_path):
        raise ValueError(f'Split file:{split_file_path} does\'nt exist!')
    with open(split_file_path, 'r') as f:
        split_config = yaml.safe_load(f)
    train_rois = [str_to_roi(roi_str) for roi_str in split_config['rois']['train']]
    val_rois = [str_to_roi(roi_str) for roi_str in split_config['rois']['val']]
    test_rois = [str_to_roi(roi_str) for roi_str in split_config['rois']['test']]
    return train_rois, val_rois, test_rois
