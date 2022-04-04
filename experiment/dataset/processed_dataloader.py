import logging
import os
from typing import Tuple, List
import numpy as np
from numpy import array
from dataset.basic_dataloader import SEN12MSCRDataset, Seasons, SEN12MSCPatchRPath, S1Bands, S2Bands, Sensor
import multiprocessing as mp
from multiprocessing import Pool
from preprocess.process import NPZ_DATA_KEY


def get_s1s2s2cloudy_processed_triplet(processed_dir: str, season: Seasons, scene_id: str,
                                       patch_id: str) -> Tuple[array]:
    triplets = []
    for sensor in Sensor:
        scene = f'{sensor.value}_{scene_id}'
        filename = f'{season.value}_{scene}_p{patch_id}.npz'
        patch_path = os.path.join(processed_dir, season.value, scene, filename)
        triplets.append(np.load(patch_path)[NPZ_DATA_KEY])
    return triplets[0], triplets[1], triplets[2]
