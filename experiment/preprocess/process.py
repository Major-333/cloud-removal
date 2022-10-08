from genericpath import exists
import logging
import os
import shutil
import random
import cv2
from typing import Tuple, List, Optional
import uuid
import traceback
import numpy as np
from numpy import array
import multiprocessing as mp
from multiprocessing import Pool
from sen12ms_cr_dataset.dataset import SEN12MSCRDataset, Season, S1Bands, S2Bands, SEN12MSCRTriplet
from preprocess.clip import dsen_clip_triplets_4d, dsen_clip_triplets_3d
from preprocess.normlize import normlize, pair_normlize
from preprocess.crop import crop_triplets_image

NPZ_DATA_KEY = 'NPZ_DATA_KEY'
LOGGING_STEP = 100
TRAIN_DIR = 'TRAIN'
TEST_DIR = 'TEST'


class Preprocess(object):

    def __init__(self,
                 origin_dir: str,
                 preprocessed_dir: str,
                 parallel_num: int = 4,
                 is_crop: bool = True,
                 convert_to_uint8: bool = False) -> None:
        self.origin_dir = origin_dir
        self.processed_dir = preprocessed_dir
        self.parallel_num = parallel_num
        self.is_crop = is_crop
        self.convert_to_uint8 = convert_to_uint8
        self.triplets = SEN12MSCRDataset(base_dir=origin_dir).get_all_triplets()

    def _scatter_triplets(self, parallel_num: Optional[int] = None) -> List[List[SEN12MSCRTriplet]]:
        if not parallel_num:
            parallel_num = self.parallel_num
        triplets = [[] for _ in range(parallel_num)]
        for index, triplet in enumerate(self.triplets):
            triplets[index % parallel_num].append(triplet)
        return triplets

    def _normlize_triplet(self, datas: Tuple[array, array, array]) -> Tuple[array, array, array]:
        s1, s2, s2cloudy = datas
        s1 = normlize(s1)
        s2, s2cloudy = pair_normlize(s2, s2cloudy)
        return s1, s2, s2cloudy

    def _clip_triplet(self, datas: Tuple[array, array, array]) -> Tuple[array, array, array]:
        return dsen_clip_triplets_3d(datas)

    def _crop_triplet(self, datas: Tuple[array, array, array]) -> Tuple[array, array, array]:
        _, h, w = datas[1].shape
        input = list(datas)
        cropped_triplets = crop_triplets_image(input, (h // 2, w // 2))
        return cropped_triplets

    def _run(self, triplets: List[SEN12MSCRTriplet]):
        try:
            for payloads_index, triplet in enumerate(triplets):
                logging.info(f'working on {payloads_index + 1} payload, progress:{payloads_index + 1}/{len(triplets)}')
                s1, s2, s2cloudy = triplet.data
                if self.is_crop:
                    s1, s2, s2cloudy = self._crop_triplet((s1, s2, s2cloudy))
                s1, s2, s2cloudy = self._clip_triplet((s1, s2, s2cloudy))
                s1, s2, s2cloudy = self._normlize_triplet((s1, s2, s2cloudy))
                if np.min(s1) < 0 or np.max(s1) > 255:
                    logging.warning(
                        f'season:{triplet.season}, scene_id:{triplet.scene_id}, patch_id:{triplet.patch_id}')
                if np.min(s2) < 0 or np.max(s2) > 255:
                    logging.warning(
                        f'season:{triplet.season}, scene_id:{triplet.scene_id}, patch_id:{triplet.patch_id}')
                if np.min(s2cloudy) < 0 or np.max(s2cloudy) > 255:
                    logging.warning(
                        f'season:{triplet.season}, scene_id:{triplet.scene_id}, patch_id:{triplet.patch_id}')
                new_triplet = SEN12MSCRTriplet(self.processed_dir,
                                               triplet.season,
                                               triplet.scene_id,
                                               triplet.patch_id,
                                               file_extension='npy')
                if not exists(self.processed_dir):
                    os.makedirs(self.processed_dir)
                new_triplet.save(self.processed_dir, (s1, s2, s2cloudy))
        except Exception as e:
            logging.error(f'Failed in Preprocess._run, error msg is:{str(e)}, trace back:{traceback.format_exc()}')

    def _debug_triplets(self, datas: Tuple[array, array, array], prefix: str):
        img_idx = 105
        prefix = f'patchid:{img_idx}_{prefix}'
        self._debug_image(datas[0][img_idx, :, :, :], prefix=f'{prefix}_s1')
        self._debug_image(datas[1][img_idx, :, :, :], prefix=f'{prefix}_s2')
        self._debug_image(datas[2][img_idx, :, :, :], prefix=f'{prefix}_s2cloudy')

    def _debug_image(self, data: array, prefix: str) -> array:
        prefix = f'pid:{os.getpid()}_{prefix}'
        if len(data.shape) != 3:
            raise ValueError(f'failed to debug image. shape error: {data.shape}')
        file_name = f'{prefix}_{uuid.uuid4().hex[:8]}.png'
        # for sar
        if data.shape[0] == 2:
            logging.info('will debug sar image')
            cv2.imwrite(file_name, data[0, :, :])
            file_name_2 = f'{prefix}_{uuid.uuid4().hex[:8]}.png'
            cv2.imwrite(file_name_2, data[1, :, :])
        # for ms image
        else:
            logging.info('will debug ms image')
            rgb_channel_idxs = list(map(lambda x: x - 1, S2Bands.RGB.value))
            rgb_img = data[rgb_channel_idxs, :, :]
            rgb_img = np.transpose(rgb_img, (1, 2, 0))
            cv2.imwrite(file_name, rgb_img)

    def multiple_run(self):
        logging.info(f'start to process, with {self.parallel_num} workers')
        triplets = self._scatter_triplets()
        with Pool(self.parallel_num) as p:
            p.map(self._run, triplets)

    def run(self):
        triplets = self._scatter_triplets(parallel_num=1)[0]
        self._run(triplets)


if __name__ == '__main__':
    logging.basicConfig(filename='logger.log',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    base_dir = '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data_v2/SEN12MS_CR'
    processed_dir = '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data_v2/PROCESSED_SEN12MS_CR'
    preprocess = Preprocess(base_dir, processed_dir, parallel_num=4)
    # needs 4 min
    preprocess.run()
