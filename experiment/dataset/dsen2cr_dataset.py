import logging
import os
import numpy as np
from typing import List, Optional
from torch.utils.data import Dataset
from dataset.basic_dataloader import SEN12MSCRDataset, Seasons, SEN12MSCPatchRPath
from dataset.processed_dataloader import get_s1s2s2cloudy_processed_triplet


class Dsen2crDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 season: Seasons,
                 scene_black_list: Optional[List[int]],
                 scene_white_list: Optional[List[int]],
                 train=True):
        if not os.path.exists(data_dir):
            raise ValueError(f'Dsen2crDataset failed to load. can not find the preprocessed data:{data_dir}')
        logging.info(f'initing the Dsen2crDataset, white list:{scene_white_list}')
        logging.info(f'initing the Dsen2crDataset, black list:{scene_black_list}')
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
                patch_path = SEN12MSCPatchRPath(season, scene_id, patch_id)
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
        return np.concatenate((s1_item, s2cloudy_item), axis=0), s2_item, {'scene_id': patch_path.scene_id, 'patch_id': patch_path.patch_id}


if __name__ == '__main__':
    logging.basicConfig(filename='logger.log',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    base_dir = '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data/SEN12MS_CR'
    dataset = Dsen2crDataset(base_dir, Seasons.SUMMER)
    input, ground_truth = dataset.__getitem__(0)
    print(input.shape, ground_truth.shape)
    print(np.sum(input[np.isnan(input)]))
