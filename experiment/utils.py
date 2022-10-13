from typing import Dict
import logging
import os
import yaml
import wandb
import torch
from torch import nn
import numpy as np
import random

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

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
