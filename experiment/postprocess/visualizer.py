import os
import cv2
import logging
import argparse
import numpy as np
from typing import Tuple, List
from utils import config_logging
from sen12ms_cr_dataset.dataset import SEN12MSCRDataset, SEN12MSCRTriplet
from runners.evaluate import PREDICTS_DIRNAME

VISUALIZATION_DIRNAME = 'visualized'
S1_VISUALIZATION_DIRNAME = 's1_visualized'
S2_VISUALIZATION_DIRNAME = 's2_visualized'
S2_CLOUDY_VISUALIZATION_DIRNAME = 's2_cloudy_visualized'
PREDICT_VISUALIZATION_DIRNAME = 'predict_visualized'

VISUALIZATION_LOGGING_NAME = 'visualized.log'

class PredictVisualizer(object):
    def __init__(self, original_dataset_path: str) -> None:
        self.original_dataset_path = original_dataset_path

    def _minmax_scale(self, s2: np.array, s2cloudy: np.array, predict: np.array) -> Tuple[np.array, np.array, np.array]:
        max_value = max(max(np.max(s2), np.max(s2cloudy)), np.max(predict))
        min_value = min(min(np.min(s2), np.min(s2cloudy)), np.min(predict))
        scaled_s2 = (s2 - min_value) / (max_value - min_value)
        scaled_s2_cloudy = (s2cloudy - min_value) / (max_value - min_value)
        scaled_predict = (predict - min_value) / (max_value - min_value)
        return scaled_s2, scaled_s2_cloudy, scaled_predict

    def _get_predict_triplets(self, predict_path: str,) -> List[SEN12MSCRTriplet]:
        predict_dataset = SEN12MSCRDataset(base_dir=predict_path, file_extension='npy')
        predict_triplets = predict_dataset.get_existed_triplets_with_x(x=PREDICTS_DIRNAME)
        return predict_triplets

    def _save_ms(self, ms: np.array, save_path: str):
        if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
        if os.path.exists(save_path):
            raise ValueError(f'patch:{save_path} has existed')
        # save visualize result for each ms
        if len(ms.shape) != 3:
            raise ValueError(f'ms shape is invalid: {ms.shape}')
        bgr_img = np.array(ms[1:4, :, :]).transpose((1, 2, 0))
        # scale to 0-255
        bgr_img = bgr_img*255
        # save
        cv2.imwrite(save_path, bgr_img)

    def _save_sar(self, sar: np.array, save_path: str):
        if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
        if os.path.exists(save_path):
            raise ValueError(f'patch:{save_path} has existed')
        # save visualize result for each ms
        if len(sar.shape) != 3 or sar.shape[0] != 2:
            raise ValueError(f'sar shape is invalid: {sar.shape}')
        empty_channel = np.expand_dims(np.zeros((sar.shape[1], sar.shape[2])), axis=0)
        bgr_img = np.concatenate((sar, empty_channel), axis=0).transpose((1, 2, 0))
        # scale to 0-255
        bgr_img = bgr_img*255
        # save
        cv2.imwrite(save_path, bgr_img)     

    def visualize_predicts(self, predict_path: str, output_dataset_path: str) -> None:
        # get triplets indexs
        predict_triplets = self._get_predict_triplets(predict_path)
        for predict_triplet in predict_triplets:
            # get predicted ms numpy arry.
            predicted_ms = predict_triplet.predict
            original_triplet = SEN12MSCRTriplet(
                dataset_dir=self.original_dataset_path,
                season=predict_triplet.season,
                scene_id=predict_triplet.scene_id,
                patch_id=predict_triplet.patch_id,
                file_extension=predict_triplet.file_extension
            )
            s1, s2, s2cloudy = original_triplet.data
            # scale for visualization
            s2, s2cloudy, predicted_ms = self._minmax_scale(s2, s2cloudy, predicted_ms)
            output_triplet = SEN12MSCRTriplet(
                dataset_dir=output_dataset_path,
                season=predict_triplet.season,
                scene_id=predict_triplet.scene_id,
                patch_id=predict_triplet.patch_id,
                file_extension=predict_triplet.file_extension
            )
            s1_save_path = output_triplet.x_path(x=S1_VISUALIZATION_DIRNAME, dataset_dir=output_dataset_path, file_extension='jpg')
            s2_save_path = output_triplet.x_path(x=S2_VISUALIZATION_DIRNAME, dataset_dir=output_dataset_path, file_extension='jpg')
            s2cloudy_save_path = output_triplet.x_path(x=S2_CLOUDY_VISUALIZATION_DIRNAME, dataset_dir=output_dataset_path, file_extension='jpg')
            predict_save_path = output_triplet.x_path(x=PREDICT_VISUALIZATION_DIRNAME, dataset_dir=output_dataset_path, file_extension='jpg')
            self._save_sar(s1, s1_save_path)
            self._save_ms(s2, s2_save_path)
            self._save_ms(s2cloudy, s2cloudy_save_path)
            self._save_ms(predicted_ms, predict_save_path)

def get_args():
    parser = argparse.ArgumentParser(description='Visualize the SAR and MS imgs')
    parser.add_argument('--original', type=str, required=True, help='Path to input dataset')
    parser.add_argument('--predict', type=str, required=True, help='Path to input dataset')
    parser.add_argument('--output', type=str, required=True, help='Path to output dataset')
    return parser.parse_args()

if __name__ == '__main__':
    # parse args
    args = get_args()
    # check output dir path
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    # init logging
    logging_file_path = os.path.join(args.output, VISUALIZATION_LOGGING_NAME)
    config_logging(filename=logging_file_path)
    # visualize
    visualizer = PredictVisualizer(args.original)
    visualizer.visualize_predicts(args.predict, args.output)
