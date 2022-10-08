import logging
import os
import numpy as np
from typing import List, Optional, Tuple
from torch.utils.data import Dataset
from sen12ms_cr_dataset.dataset import Season, SEN12MSCRPatchPath, Sensor
from preprocess.process import NPZ_DATA_KEY


def get_s1s2s2cloudy_processed_triplet(processed_dir: str, season: Season, scene_id: str,
                                       patch_id: str) -> Tuple[np.array]:
    triplets = []
    for sensor in Sensor:
        scene = f'{sensor.value}_{scene_id}'
        filename = f'{season.value}_{scene}_p{patch_id}.npz'
        patch_path = os.path.join(processed_dir, season.value, scene, filename)
        triplets.append(np.load(patch_path)[NPZ_DATA_KEY])
    return triplets[0], triplets[1], triplets[2]


class ProcessedSEN12MSCRDataset(Dataset):

    def __init__(self, data_dir: str, season: Season):
        if not os.path.exists(data_dir):
            raise ValueError(f'Dsen2crDataset failed to load. can not find the preprocessed data:{data_dir}')
        self.data_dir = data_dir
        self.patch_paths = []
        season_path = os.path.join(self.data_dir, season.value)
        scene_dir_names = os.listdir(season_path)
        scene_ids = list(set([name.split('_')[-1] for name in scene_dir_names]))
        for scene_id in scene_ids:
            s1_scene_path = os.path.join(season_path, f's1_{scene_id}')
            filenames = os.listdir(s1_scene_path)
            patch_ids = [filename.split('.')[0].split('p')[-1] for filename in filenames]
            for patch_id in patch_ids:
                patch_path = SEN12MSCRPatchPath(season, scene_id, patch_id)
                self.patch_paths.append(patch_path)
        logging.info(f'dataset init finished, scene_ids:{scene_ids}, size:{len(self.patch_paths)}')
        # HACK: for quick test
        # self.patch_paths = self.patch_paths[:200]

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, index):
        patch_path = self.patch_paths[index]
        s1_item, s2_item, s2cloudy_item = get_s1s2s2cloudy_processed_triplet(self.data_dir, patch_path.season,
                                                                             patch_path.scene_id, patch_path.patch_id)
        return np.concatenate((s1_item, s2cloudy_item), axis=0), s2_item, {
            'scene_id': patch_path.scene_id,
            'patch_id': patch_path.patch_id
        }


if __name__ == '__main__':
    logging.basicConfig(filename='logger.log',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    base_dir = '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data/SEN12MS_CR_V4'
    dataset = ProcessedSEN12MSCRDataset(base_dir, Season.SUMMER)
    input, ground_truth, info = dataset.__getitem__(0)
    print(input.shape, ground_truth.shape)
    print(np.sum(input[np.isnan(input)]))
