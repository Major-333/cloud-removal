import os
from tqdm import tqdm
import logging
import random
from typing import List
import traceback
import numpy as np
from sen12ms_cr_dataset.dataset import SEN12MSCRDataset, Sensor, SEN12MSCRTriplet
from sen12ms_cr_dataset.feature_detectors import get_cloud_cloudshadow_mask

NPZ_DATA_KEY = 'NPZ_DATA_KEY'
LOGGING_STEP = 100
TRAIN_DIR = 'TRAIN'
TEST_DIR = 'TEST'
RAW_SIZE = 256

class Preprocess(object):
    def __init__(self, origin_dir: str, preprocessed_dir: str, crop_size: int, raw_size: int, use_cloud_mask: bool, cloud_threshold: float=0.2) -> None:
        self.origin_dir = origin_dir
        self.preprocessed_dir = preprocessed_dir
        self.crop_size = crop_size
        self.raw_size = raw_size
        self.use_cloud_mask = use_cloud_mask
        if use_cloud_mask:
            self.cloud_threshold = cloud_threshold
        self.triplets = SEN12MSCRDataset(base_dir=origin_dir).get_all_triplets()
        self.clip_lower_bound = {
            Sensor.s1: [-25.0, -32.5],
            Sensor.s2: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            Sensor.s2cloudy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
        self.clip_upper_bound = {
            Sensor.s1: [0, 0],
            Sensor.s2: [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
            Sensor.s2cloudy: [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
        }
        self.max_value = 1
        self.scale = 10000.0

    def _normalize(self, img: np.array, sensor: Sensor):
        if sensor == Sensor.s1:
            for channel_idx in range(len(img)):
                img[channel_idx] = np.clip(img[channel_idx], self.clip_lower_bound[sensor][channel_idx], self.clip_upper_bound[sensor][channel_idx])
                img[channel_idx] -= self.clip_lower_bound[sensor][channel_idx]
                img[channel_idx] = self.max_value * (img[channel_idx] / (self.clip_upper_bound[sensor][channel_idx] - self.clip_lower_bound[sensor][channel_idx]))
        elif sensor == Sensor.s2 or sensor == Sensor.s2cloudy:
            for channel_idx in range(len(img)):
                img[channel_idx] = np.clip(img[channel_idx], self.clip_lower_bound[sensor][channel_idx], self.clip_upper_bound[sensor][channel_idx])
            # TODO: will remove it
            img /= self.scale
        else: 
            raise ValueError(f'sensor:{sensor} is invalid')
        return img
    
    def _crop_imgs(self, imgs: List[np.array]) -> List[np.array]:
        left_top_h = random.randint(0, np.maximum(0, self.raw_size - self.crop_size))
        left_top_w = random.randint(0, np.maximum(0, self.raw_size - self.crop_size))
        cropped_imgs = []
        for img in imgs:
            if len(img.shape) == 3:
                img = img[:, left_top_h:left_top_h + self.crop_size, left_top_w:left_top_w + self.crop_size]
            elif len(img.shape) == 2:
                img = img[left_top_h:left_top_h + self.crop_size, left_top_w:left_top_w + self.crop_size]
            else:
                raise ValueError(f'input image shape is:{img.shape}')
            cropped_imgs.append(img)
        return cropped_imgs

    def run(self):
        for triplet in tqdm(self.triplets):
            s1, s2, s2cloudy = triplet.data
            s1, s2, s2cloudy = np.float32(s1), np.float32(s2), np.float32(s2cloudy)
            if self.use_cloud_mask:
                cloud_mask = get_cloud_cloudshadow_mask(s2cloudy, self.cloud_threshold)
                cloud_mask[cloud_mask != 0] = 1
            s1 = self._normalize(s1, Sensor.s1)
            s2 = self._normalize(s2, Sensor.s2)
            s2cloudy = self._normalize(s2cloudy, Sensor.s2cloudy)
            if self.use_cloud_mask:
                s1, s2, s2cloudy, cloud_mask = self._crop_imgs([s1, s2, s2cloudy, cloud_mask])
            else:
                s1, s2, s2cloudy = self._crop_imgs([s1, s2, s2cloudy])
            new_triplet = SEN12MSCRTriplet(self.preprocessed_dir,
                                            triplet.season,
                                            triplet.scene_id,
                                            triplet.patch_id,
                                            file_extension='npy')
            if not os.path.exists(self.preprocessed_dir):
                os.makedirs(self.preprocessed_dir)
            if self.use_cloud_mask:
                new_triplet.save_with_cloudy_mask(self.preprocessed_dir, data=(s1, s2, s2cloudy, cloud_mask))
            else:
                new_triplet.save(self.preprocessed_dir, data=(s1, s2, s2cloudy))


if __name__ == '__main__':
    logging.basicConfig(filename='logger.log',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    base_dir = '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data_v2/SEN12MS_CR'
    processed_dir = '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data_v2/PROCESSED_SEN12MS_CR_V4'
    # preprocess = Preprocess(base_dir, processed_dir, crop_size=128, raw_size=256, use_cloud_mask=True, cloud_threshold=0.2)
    preprocess = Preprocess(base_dir, processed_dir, crop_size=128, raw_size=256, use_cloud_mask=True)

    # needs 4 min
    preprocess.run()
