import logging
import os
import numpy as np
from typing import List, Optional
from torch.utils.data import Dataset
from dataset.basic_dataloader import SEN12MSCRDataset, Seasons, SEN12MSCPatchRPath
from dataset.processed_dataloader import get_s1s2s2cloudy_processed_triplet


class Dsen2crDataset(Dataset):

    def __init__(self,
                 base_dir: str,
                 processed_dir: str,
                 season: Seasons,
                 scene_black_list: Optional[List[int]],
                 scene_white_list: Optional[List[int]],
                 train=True):
        logging.info(f'initing the Dsen2crDataset, white list:{scene_white_list}')
        logging.info(f'initing the Dsen2crDataset, black list:{scene_black_list}')
        self.processed_dir = processed_dir
        self.dataloader = SEN12MSCRDataset(base_dir)
        self.patch_paths = []
        scene_ids = self.dataloader.get_season_ids(Seasons.SUMMER.value)
        for scene_id in scene_ids.keys():
            # skip the scene which is in the black list
            if scene_black_list and int(scene_id) in scene_black_list:
                continue
            # skip the scene which is not in the white list
            if scene_white_list and int(scene_id) not in scene_white_list:
                continue
            patch_ids = scene_ids[scene_id]
            for patch_id in patch_ids:
                patch_path = SEN12MSCPatchRPath(season, scene_id, patch_id)
                self.patch_paths.append(patch_path)
        logging.info(f'scene_ids:{list(scene_ids.keys())}')
        if not os.path.exists(processed_dir):
            raise ValueError(f'Dsen2crDataset failed to load. can not find the preprocessed data:{processed_dir}')
        # HACK: for quick test
        # self.patch_paths = self.patch_paths[:200]

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, index):
        patch_path = self.patch_paths[index]
        s1_item, s2_item, s2cloudy_item = get_s1s2s2cloudy_processed_triplet(self.processed_dir, patch_path.season,
                                                                             patch_path.scene_id, patch_path.patch_id)
        return np.concatenate((s1_item, s2cloudy_item), axis=0), s2_item


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
