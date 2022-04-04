import logging
import os
from typing import Tuple, List, Optional
import numpy as np
from numpy import array
import multiprocessing as mp
from multiprocessing import Pool
from dataset.basic_dataloader import SEN12MSCRDataset, Seasons, SEN12MSCPatchRPath, S1Bands, S2Bands, Sensor
from preprocess.clip import dsen_clip_triplets
from preprocess.normlize import normlize_to_uint8_foreach_channel, normlize_to_uint8
from preprocess.crop import crop_triplets_image

NPZ_DATA_KEY = 'NPZ_DATA_KEY'
LOGGING_STEP = 100


class PreprocessParam(object):

    def __init__(self, scene_id: str, patch_ids: List[str]):
        self.scene_id = scene_id
        self.patch_ids = patch_ids


class Preprocess(object):

    def __init__(self,
                 origin_dir: str,
                 preprocessed_dir: str,
                 parallel_num: int = 4,
                 overwrite: bool = True,
                 season: Seasons = Seasons.SUMMER,
                 is_scale_foreach_channel: bool = True) -> None:
        self.origin_dir = origin_dir
        self.preprocessed_dir = preprocessed_dir
        self.parallel_num = parallel_num
        self.overwrite = overwrite
        self.season = season
        self.is_normlize_foreach_channel = is_scale_foreach_channel
        self.dataloader = SEN12MSCRDataset(base_dir=origin_dir)

    def _scatter_payloads(self, parallel_num: Optional[int] = None) -> List[List[PreprocessParam]]:
        if not parallel_num:
            parallel_num = self.parallel_num
        scene_ids = self.dataloader.get_season_ids(self.season.value)
        params = [[] for _ in range(parallel_num)]
        for index, scene_id in enumerate(scene_ids):
            params[index % self.parallel_num].append(PreprocessParam(scene_id, scene_ids[scene_id]))
        return params

    def _get_triplets_by_scene(self, scene_id: int):
        s1, s2, s2cloudy, _ = self.dataloader.get_triplets(self.season,
                                                           scene_ids=scene_id,
                                                           s1_bands=S1Bands.ALL,
                                                           s2_bands=S2Bands.ALL,
                                                           s2cloudy_bands=S2Bands.ALL)
        return s1.astype('float32'), s2.astype('float32'), s2cloudy.astype('float32')

    def _normlize_triplets_to_uint8(self, datas: Tuple[array, array, array]) -> Tuple[array, array, array]:
        logging.info('will normlize the triplets')
        if self.is_normlize_foreach_channel:
            return tuple([normlize_to_uint8_foreach_channel(datas[i]) for i in range(3)])

        else:
            return tuple([normlize_to_uint8(datas[i]) for i in range(3)])

    def _clip_triplets(self, datas: Tuple[array, array, array]) -> Tuple[array, array, array]:
        logging.info('will clip the triplets')
        return dsen_clip_triplets(datas)

    def _crop_triplets(self, datas: Tuple[array, array, array]) -> Tuple[array, array, array]:
        logging.info('will crop the triplets')
        sar_channel = datas[0].shape[1]
        n, c, h, w = datas[1].shape
        cropped_images = [np.zeros((n, sar_channel, h//2, w//2))] + \
            [np.zeros((n, c, h//2, w//2)) for _ in range(2)]
        # TODO: optimize this for-loop
        for image_idx in range(n):
            input = [data[image_idx, :, :, :] for data in datas]
            cropped_triplets = crop_triplets_image(input, (h // 2, w // 2))
            for i in range(3):
                cropped_images[i][image_idx, :, :, :] = cropped_triplets[i]
        return tuple(cropped_images)

    def _save_triplets(self, datas: Tuple[array, array, array], param: PreprocessParam) -> Tuple[array, array, array]:
        scene_id = param.scene_id
        patch_ids = param.patch_ids
        for index, patch_id in enumerate(patch_ids):
            if index % LOGGING_STEP == 0:
                logging.info(f'scene_id:{scene_id}: processing on index:{index}')
            for sensor_id, sensor in enumerate(Sensor):
                data = datas[sensor_id]
                scene = f'{sensor.value}_{scene_id}'
                filename = f'{self.season.value}_{scene}_p{patch_id}'
                patch_dir = os.path.join(self.preprocessed_dir, self.season.value, scene)
                if not os.path.exists(patch_dir):
                    os.makedirs(patch_dir)
                patch_path = os.path.join(patch_dir, filename)
                patch = data[index, :, :, :]
                if os.path.exists(patch_path) and not self.overwrite:
                    continue
                else:
                    np.savez_compressed(patch_path, NPZ_DATA_KEY=patch)

    def _run(self, params: List[PreprocessParam]):
        try:
            for payloads_index, param in enumerate(params):
                logging.info(f'working on {payloads_index + 1} payload, progress:{payloads_index + 1}/{len(params)}')
                scene_id = param.scene_id
                s1, s2, s2cloudy = self._get_triplets_by_scene(scene_id)
                s1, s2, s2cloudy = self._crop_triplets((s1, s2, s2cloudy))
                s1, s2, s2cloudy = self._clip_triplets((s1, s2, s2cloudy))
                s1, s2, s2cloudy = self._normlize_triplets_to_uint8((s1, s2, s2cloudy))
                logging.info('process finished, will save the files')
                self._save_triplets((s1, s2, s2cloudy), param)
                logging.info('processed data has been saved')
        except Exception as e:
            logging.error(f'Failed in Preprocess._run, error msg is:{str(e)}')

    def multiple_run(self):
        logging.info(f'start to process, with {self.parallel_num} workers')
        params = self._scatter_payloads()
        with Pool(self.parallel_num) as p:
            p.map(self._run, params)

    def run(self):
        params = self._scatter_payloads(parallel_num=1)[0]
        self._run(params)


if __name__ == '__main__':
    logging.basicConfig(filename='logger.log',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    base_dir = '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data/SEN12MS_CR'
    processed_dir = '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data/SEN12MS_CR_PROCESSED_V2'
    preprocess = Preprocess(base_dir, processed_dir, 4, True)
    # needs 4 min
    preprocess.multiple_run()