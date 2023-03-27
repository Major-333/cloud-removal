import os
import shutil
from typing import Optional, Dict, List
from utils import increment_path, setup_seed, config_logging, DEFAULT_LOG_FILENAME

CONFIG_FILENAME = 'config-defaults.yaml'
DEFAULT_SPLIT_FILENAME = 'split.yaml'
EXP_SUBDIR_NAME = 'exp'

class Runner(object):
    def __init__(self, config: Dict, save_subdir_name: str) -> None:
        self._runner_init(config, save_subdir_name)

    def _runner_init(self, config: Dict, save_subdir_name: str) -> None:
        # Load config to trainer
        self._parse_config(config)
        self.config = config
        # Fix random seed for reproducibility
        setup_seed(self.seed)
        # for save.
        self.save_dir = self._get_save_dir(save_subdir_name)
        # init logging
        logging_file_path = os.path.join(self.save_dir, DEFAULT_LOG_FILENAME)
        config_logging(filename=logging_file_path)

    def _get_save_dir(self, save_subdir_name: str) -> str:
        save_dir = os.path.join(self.save_dir, save_subdir_name, EXP_SUBDIR_NAME)
        exp_dir = increment_path(save_dir)
        # save metadata info
        config_path = os.path.join(exp_dir, CONFIG_FILENAME)
        shutil.copyfile(CONFIG_FILENAME, config_path)
        if self.split_file_path:
            split_file_path = os.path.join(exp_dir, DEFAULT_SPLIT_FILENAME)
            shutil.copyfile(self.split_file_path, split_file_path)
        return exp_dir

    def _parse_config(self, config: Dict):
        self._parse_general_config(config)
        self._parse_model_specific_config(config)
    
    def _parse_general_config(self, config: Dict):
        self.max_epoch = config['epochs']
        self.dataset_path = config['dataset']
        self.batch_size = config['batch_size']
        self.validate_every = config['validate_every']
        self.save_dir = config['save_dir']
        self.dataset_file_extension = config['dataset_file_extension']
        self.seed = config['seed']
        self.use_cloud_mask = config['use_cloud_mask']
        self.debug = config['debug']
        self.split_file_path = config.get('split_file_path')
        
    def _parse_model_specific_config(self, config: Dict):
        self.model_name = config['model']
        self.lr = config['lr']
        self.min_lr = config['min_lr']
        self.loss_name = config['loss_fn']