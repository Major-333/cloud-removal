import numpy as np
from numpy import array
from typing import Tuple


def iqr_clip_foreach_channel(data: array, factor: float = 6.0) -> array:
    if len(data.shape) != 4:
        raise ValueError(f'data input should be: N,C,H,W, but got shape:{data.shape}')
    total_channels = data.shape[1]
    for index in range(total_channels):
        data_slice = data[:, index, :, :]
        q75, q25 = np.percentile(data_slice.reshape(-1), [75, 25])
        iqr = q75 - q25
        low_bound = q25 - factor * iqr
        up_bound = q75 + factor * iqr
        data_slice[data_slice > up_bound] = up_bound
        data_slice[data_slice < low_bound] = low_bound
    return data


def dsen_clip_triplets_4d(datas: Tuple[array, array, array]) -> Tuple[array, array, array]:
    if len(datas[0].shape) != 4 or len(datas[1].shape) != 4 or len(datas[2].shape) != 4:
        raise ValueError(f'data input should be: N,C,H,W, but got shape:{datas[0].shape} and {datas[1].shape} and {datas[2].shape}')
    sar0 = datas[0][:, 0, :, :]
    sar1 = datas[0][:, 1, :, :]
    sar0[sar0 < -25.0] = -25.0
    sar0[sar0 > 0] = 0
    sar1[sar1 < -32.5] = -32.5
    sar1[sar1 > 0] = 0
    datas[1][datas[1] < 0] = 0
    datas[1][datas[1] > 10000] = 10000
    datas[2][datas[2] < 0] = 0
    datas[2][datas[2] > 10000] = 10000
    return datas

def dsen_clip_triplets_3d(datas: Tuple[array, array, array]) -> Tuple[array, array, array]:
    if len(datas[0].shape) != 3 or len(datas[1].shape) != 3 or len(datas[2].shape) != 3:
        raise ValueError(f'data input should be: C,H,W, but got shape:{datas[0].shape} and {datas[1].shape} and {datas[2].shape}')
    sar0 = datas[0][0, :, :]
    sar1 = datas[0][1, :, :]
    sar0[sar0 < -25.0] = -25.0
    sar0[sar0 > 0] = 0
    sar1[sar1 < -32.5] = -32.5
    sar1[sar1 > 0] = 0
    datas[1][datas[1] < 0] = 0
    datas[1][datas[1] > 10000] = 10000
    datas[2][datas[2] < 0] = 0
    datas[2][datas[2] > 10000] = 10000
    return datas