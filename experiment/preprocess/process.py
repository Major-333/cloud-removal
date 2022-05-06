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
from dataset.basic_dataloader import SEN12MSCRDataset, Seasons, SEN12MSCPatchRPath, S1Bands, S2Bands, Sensor
from preprocess.clip import dsen_clip_triplets_4d, dsen_clip_triplets_3d
from preprocess.normlize import normlize, pair_normlize
from preprocess.crop import crop_triplets_image

NPZ_DATA_KEY = 'NPZ_DATA_KEY'
LOGGING_STEP = 100
TRAIN_DIR = 'TRAIN'
TEST_DIR = 'TEST'


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
                 is_crop: bool = True,
                 convert_to_uint8: bool = False) -> None:
        self.origin_dir = origin_dir
        self.preprocessed_dir = preprocessed_dir
        self.parallel_num = parallel_num
        self.overwrite = overwrite
        self.season = season
        self.is_crop = is_crop
        self.convert_to_uint8 = convert_to_uint8
        self.dataloader = SEN12MSCRDataset(base_dir=origin_dir)

    def _scatter_payloads(self, parallel_num: Optional[int] = None) -> List[List[PreprocessParam]]:
        if not parallel_num:
            parallel_num = self.parallel_num
        scene_ids = self.dataloader.get_season_ids(self.season.value)
        params = [[] for _ in range(parallel_num)]
        for index, scene_id in enumerate(scene_ids):
            params[index % self.parallel_num].append(PreprocessParam(scene_id, scene_ids[scene_id]))
        return params

    def _get_triplets_by_patch(self, scene_id: int, patch_id: int):
        s1, s2, s2cloudy, _ = self.dataloader.get_s1s2s2cloudy_triplet(self.season, scene_id, patch_id, S1Bands.ALL, S2Bands.ALL, S2Bands.ALL)
        return s1.astype('float32'), s2.astype('float32'), s2cloudy.astype('float32')

    def _normlize_triplets_by_patch(self, datas: Tuple[array, array, array]) -> Tuple[array, array, array]:
        s1, s2, s2cloudy = datas
        s1 = normlize(s1)
        s2, s2cloudy = pair_normlize(s2, s2cloudy)
        return s1, s2, s2cloudy

    def _clip_triplets_by_patch(self, datas: Tuple[array, array, array]) -> Tuple[array, array, array]:
        return dsen_clip_triplets_3d(datas)

    def _crop_triplets_by_patch(self, datas: Tuple[array, array, array]) -> Tuple[array, array, array]:
        _, h, w = datas[1].shape
        input = list(datas)
        cropped_triplets = crop_triplets_image(input, (h // 2, w // 2))
        return cropped_triplets

    def _save_triplets_by_patch(self, datas: Tuple[array, array, array], scene_id: str, patch_id: int) -> Tuple[array, array, array]:
        if patch_id % LOGGING_STEP == 0:
            logging.info(f'scene_id:{scene_id}: processing on index:{patch_id}')
        for sensor_id, sensor in enumerate(Sensor):
            data = datas[sensor_id]
            scene = f'{sensor.value}_{scene_id}'
            filename = f'{self.season.value}_{scene}_p{patch_id}'
            patch_dir = os.path.join(self.preprocessed_dir, self.season.value, scene)
            if not os.path.exists(patch_dir):
                os.makedirs(patch_dir)
            patch_path = os.path.join(patch_dir, filename)
            if os.path.exists(patch_path) and not self.overwrite:
                continue
            else:
                np.savez_compressed(patch_path, NPZ_DATA_KEY=data)

    def _copy_files(self, paths_list: List[str], target_dir: str):
        for triplets_path in paths_list:
            for rel_path in triplets_path:
                processed_path = os.path.join(self.preprocessed_dir, rel_path)
                target_path = os.path.join(target_dir, rel_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copyfile(processed_path, target_path)

    def _split_to_train_validata_test(self, param: PreprocessParam):
        """ train: test = 6:2
        """
        scene_id = param.scene_id
        patch_ids = param.patch_ids    
        test_path = os.path.join(self.preprocessed_dir, TEST_DIR)
        triplets_paths = []
        for patch_id in patch_ids:
            patch_rel_path_triplets = []
            for sensor in Sensor:
                scene = f'{sensor.value}_{scene_id}'
                filename = f'{self.season.value}_{scene}_p{patch_id}.npz'
                patch_rel_path = os.path.join(self.season.value, scene, filename)
                patch_rel_path_triplets.append(patch_rel_path)
            triplets_paths.append(patch_rel_path_triplets)
        # copy file
        random.shuffle(triplets_paths)
        total = len(triplets_paths)
        train_list = triplets_paths[:int(total*0.8)]
        test_list = triplets_paths[int(total*0.8):]
        train_path = os.path.join(self.preprocessed_dir, TRAIN_DIR)
        test_path = os.path.join(self.preprocessed_dir, TEST_DIR)
        self._copy_files(train_list, train_path)
        self._copy_files(test_list, test_path)


    def _run(self, params: List[PreprocessParam]):
        try:
            for payloads_index, param in enumerate(params):
                logging.info(f'working on {payloads_index + 1} payload, progress:{payloads_index + 1}/{len(params)}')
                scene_id = param.scene_id
                patch_ids = param.patch_ids
                for patch_id in patch_ids:
                    s1, s2, s2cloudy = self._get_triplets_by_patch(scene_id, patch_id)
                    if self.is_crop:
                        s1, s2, s2cloudy = self._crop_triplets_by_patch((s1, s2, s2cloudy))
                    s1, s2, s2cloudy = self._clip_triplets_by_patch((s1, s2, s2cloudy))
                    s1, s2, s2cloudy = self._normlize_triplets_by_patch((s1, s2, s2cloudy))
                    if self.convert_to_uint8:
                        s1, s2, s2cloudy = s1.astype('uint8'), s2.astype('uint8'), s2cloudy.astype('uint8')
                    self._save_triplets_by_patch((s1, s2, s2cloudy), scene_id, patch_id)
                logging.info(f'scene:{scene_id} processed data has been saved')
                self._split_to_train_validata_test(param)
                logging.info('split finished')
                # # old
                # s1, s2, s2cloudy = self._get_triplets_by_scene(scene_id)
                # if self.is_crop:
                #     s1, s2, s2cloudy = self._crop_triplets_by_scene((s1, s2, s2cloudy))
                
                # s1, s2, s2cloudy = self._clip_triplets_by_scene((s1, s2, s2cloudy))
                # s1, s2, s2cloudy = self._normlize_triplets((s1, s2, s2cloudy))     
                # if self.convert_to_uint8:
                #     s1, s2, s2cloudy = s1.astype('uint8'), s2.astype('uint8'), s2cloudy.astype('uint8')
                # logging.info('process finished, will save the files')
                # self._save_triplets((s1, s2, s2cloudy), param)
                # logging.info('processed data has been saved')
                # self._split_to_train_validata_test(param)
                # logging.info('split finished')
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
    processed_dir = '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data/SEN12MS_CR_PROCESSED_V3'
    preprocess = Preprocess(base_dir, processed_dir, parallel_num=4, overwrite=True, is_scale_foreach_channel=True)
    # needs 4 min
    preprocess.multiple_run()
