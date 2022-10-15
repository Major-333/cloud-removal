import yaml
import argparse
import logging
import numpy as np
import random
from typing import List, Tuple
from utils import config_logging, setup_seed, roi_to_str, str_to_roi
from sen12ms_cr_dataset.dataset import Roi, SEN12MSCRDataset, SEN12MSCRTriplet, Season
from train import DEFAULT_SPLIT_FILENAME

def run(dataset_path: str) -> None:
    dataset = SEN12MSCRDataset(dataset_path,file_extension='npy')
    triplets = dataset.triplets
    season_roi_strs = {
        Season.SPRING.value: [],
        Season.SUMMER.value: [],
        Season.FALL.value: [],
        Season.WINTER.value: [],
    }
    total_counts = 0
    for triplet in triplets:
        season_roi_strs[triplet.roi.season.value].append(roi_to_str(triplet.roi))
    for key in season_roi_strs:
        season_roi_strs[key] = sorted(set(season_roi_strs[key]))
        random.shuffle(season_roi_strs[key])
        total_counts += len(season_roi_strs[key])
    logging.info(f'total roi counts is:{total_counts}')
    # count roi by seasons
    season_counts = {k:len(v) for k, v in season_roi_strs.items()}
    logging.info(f'roi counts by season:{season_counts}')
    season_counts = {k:round(v*10/total_counts) for k, v in season_counts.items()}
    logging.info(f'roi counts by season in 10 roi:{season_counts}')
    # split by season
    selected_season_roi_strs = {k:np.random.choice(v, size=season_counts[k]*2, replace=False).tolist() for k, v in season_roi_strs.items()}
    test_season_roi_strs = {k:v[:season_counts[k]] for k, v in selected_season_roi_strs.items()}
    val_season_roi_strs = {k:v[season_counts[k]:] for k, v in selected_season_roi_strs.items()}
    logging.info(f'val rois:{val_season_roi_strs}')
    logging.info(f'test rois:{test_season_roi_strs}')
    # get train rois
    train_season_roi_strs = {}
    for season_value in season_roi_strs:
        roi_strs = season_roi_strs[season_value]
        train_roi_strs = sorted(set(roi_strs) - set(val_season_roi_strs[season_value]) - set(test_season_roi_strs[season_value]))
        random.shuffle(train_roi_strs)
        train_season_roi_strs[season_value] = train_roi_strs
    # rois
    train_roi_strs = []
    val_roi_strs = []
    test_roi_strs = []
    for season_value in season_counts:
        train_roi_strs.extend(train_season_roi_strs[season_value])
        val_roi_strs.extend(val_season_roi_strs[season_value])
        test_roi_strs.extend(test_season_roi_strs[season_value])
    # save to yaml
    split_config = {
        'metadata': {
            'total_counts': total_counts,
            'all_counts': {k:len(v) for k, v in season_roi_strs.items()},
            'train_counts': {
                Season.SPRING.value: len(train_season_roi_strs[Season.SPRING.value]),
                Season.SUMMER.value: len(train_season_roi_strs[Season.SUMMER.value]),
                Season.FALL.value: len(train_season_roi_strs[Season.FALL.value]),
                Season.WINTER.value: len(train_season_roi_strs[Season.WINTER.value]),
            },
            'val_counts': {
                Season.SPRING.value: len(val_season_roi_strs[Season.SPRING.value]),
                Season.SUMMER.value: len(val_season_roi_strs[Season.SUMMER.value]),
                Season.FALL.value: len(val_season_roi_strs[Season.FALL.value]),
                Season.WINTER.value: len(val_season_roi_strs[Season.WINTER.value]),
            },
            'test_counts': {
                Season.SPRING.value: len(test_season_roi_strs[Season.SPRING.value]),
                Season.SUMMER.value: len(test_season_roi_strs[Season.SUMMER.value]),
                Season.FALL.value: len(test_season_roi_strs[Season.FALL.value]),
                Season.WINTER.value: len(test_season_roi_strs[Season.WINTER.value]),
            }
        },
        'rois': {
            'train': train_roi_strs,
            'val': val_roi_strs,
            'test': test_roi_strs,
        },
    }
    with open(DEFAULT_SPLIT_FILENAME, 'w') as wf:
        yaml.dump(split_config, wf, sort_keys=False, default_flow_style=False)
        

def get_args():
    parser = argparse.ArgumentParser(description='Train the Cloud Remove network')
    parser.add_argument('--dataset', type=str, help='Dataset dir path')
    parser.add_argument('--seed', type=str, help='Fix random seed for reproducibility')
    return parser.parse_args()

if __name__ == '__main__':
    config_logging()
    args = get_args()
    setup_seed(int(args.seed))
    run(args.dataset)