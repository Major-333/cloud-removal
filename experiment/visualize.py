import os
import torch
import numpy as np
import logging
from experiment.sen12ms_cr_dataset.dataset import Season, S2Bands
from sen12ms_cr_dataset.visualize import save_patch
from models.mprnet import MPRNet
from init_helper import InitHelper
from typing import Optional, Dict, Tuple
from utils import load_ddp_checkpoint, parse_wandb_yaml
import argparse

CONFIG_FILEPATH = './config-defaults.yaml'


class Visualizer(object):

    def __init__(self, config: Dict, device: str, checkpoint_path: str) -> None:
        self._config = config
        if self._config['model'] != os.path.basename(os.path.dirname(checkpoint_path)):
            raise ValueError('model name conflict!')
        self._device = device
        self._init_helper = InitHelper(self._config)
        model = InitHelper(self._config).init_model()
        logging.info(f'will evaluate the checkpoint:{checkpoint_path}')
        self._model = load_ddp_checkpoint(model, checkpoint_path, self._device)

    def visualize_on_test_by_patch(self, patch_id: str, scene_id: str):
        model = self._model
        sar_patch, ground_truth, s2cloudy_patch = get_s1s2s2cloudy_processed_triplet(self._config['test_dir'],
                                                                                     Season.SUMMER,
                                                                                     scene_id=scene_id,
                                                                                     patch_id=patch_id)
        base_dir = os.path.join(self._config['visual_perception_dir'], self._config['model'])
        sar_filepath = os.path.join(base_dir, f'SAR-{scene_id}-{patch_id}.png')
        input_filepath = os.path.join(base_dir, f'input-{scene_id}-{patch_id}.png')
        output_filepath = os.path.join(base_dir, f'output-{scene_id}-{patch_id}.png')
        ground_truth_filepath = os.path.join(base_dir, f'groundtruth-{scene_id}-{patch_id}.png')
        rgb_channel_idxs = list(map(lambda x: x - 1, S2Bands.RGB.value))
        ground_truth_rgb_patch = ground_truth[rgb_channel_idxs, :, :]
        input_rgb_patch = s2cloudy_patch[rgb_channel_idxs, :, :]
        cloudy_input = np.concatenate((sar_patch, s2cloudy_patch), axis=0)
        cloudy_input = np.expand_dims(cloudy_input, axis=0)
        cloudy_input = torch.tensor(cloudy_input)
        cloudy_input = cloudy_input.cuda()
        with torch.no_grad():
            if isinstance(model, MPRNet):
                output = model(cloudy_input)[2]
            else:
                output = model(cloudy_input)
            output[output < 0] = 0
            output[output > 255] = 255
        output_rgb_patch = output.cpu().detach().numpy()[0, rgb_channel_idxs, :, :]
        save_patch(sar_patch, sar_filepath)
        save_patch(input_rgb_patch, input_filepath)
        save_patch(ground_truth_rgb_patch, ground_truth_filepath)
        save_patch(output_rgb_patch, output_filepath)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', '-f', type=str, required=True, help='Load model from a .pth file')
    parser.add_argument('--scene_id', '-s', type=str, required=True)
    parser.add_argument('--patch_id', '-p', type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(filename='logger.log',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    args = get_args()
    if not os.getenv('CUDA_VISIBLE_DEVICES'):
        raise ValueError(f'set the env: `CUDA_VISIBLE_DEVICES` first')
    if not os.path.exists('config-defaults.yaml'):
        raise ValueError(f'can not find the config file: config-defaults.yaml')
    config = parse_wandb_yaml(CONFIG_FILEPATH)
    print(f'config:\n{config}')
    checkpoint_relpath = args.load
    checkpoint_path = os.path.join(config['checkpoints_dir'], checkpoint_relpath)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'will use gpu:{device}')
    visualizer = Visualizer(config, device, checkpoint_path)
    visualizer.visualize_on_test_by_patch(args.patch_id, args.scene_id)
