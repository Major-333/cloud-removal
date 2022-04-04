import random
from matplotlib.pyplot import axis
import numpy as np
from numpy import array
from typing import Tuple, List


def crop_triplets_image(image3ds: List[array], output_size: Tuple[int, int]) -> List[array]:
    if len(image3ds[0].shape) != 3 or len(image3ds[1].shape) != 3 or len(image3ds[2].shape) != 3:
        # image shape is: C H W
        raise ValueError(
            f'failed to image crop: image shape is invalid:{image3ds[0].shape}, {image3ds[1].shape}, {image3ds[2].shape}'
        )
    in_h, in_w = image3ds[0].shape[1], image3ds[0].shape[2]
    out_h, out_w = output_size
    if in_h < out_h or in_w < out_w:
        raise ValueError(
            f'failed to image crop: output size is bigger than raw size. raw size:{image3ds[0].shape}, output size:{output_size}'
        )
    left_top_h = random.randint(0, in_h - out_h)
    left_top_w = random.randint(0, in_w - out_w)
    cropped_images = [img[:, left_top_h:left_top_h + out_h, left_top_w:left_top_w + out_w] for img in image3ds]
    return cropped_images

