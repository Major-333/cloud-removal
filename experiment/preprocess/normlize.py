import numpy as np
from numpy import array


def minmax_scaler_normlize_foreach_channel(data: array) -> array:
    if len(data.shape) != 4:
        raise ValueError(f'data input should be: N,C,H,W, but got shape:{data.shape}')
    total_channels = data.shape[1]
    for index in range(total_channels):
        data_slice = data[:, index, :, :]
        data[:, index, :, :] = (data_slice - np.min(data_slice)) / \
            (np.max(data_slice) - np.min(data_slice))
    return data


def minmax_scaler_normlize(data: array) -> array:
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def normlize_foreach_channel(data: array) -> array:
    if len(data.shape) != 4:
        raise ValueError(f'data input should be: N,C,H,W, but got shape:{data.shape}')
    total_channels = data.shape[1]
    for index in range(total_channels):
        data_slice = data[:, index, :, :]
        data[:, index, :, :] = (data_slice - np.min(data_slice)) / (np.max(data_slice) - np.min(data_slice) + 1e-10) * 255
    return data


def normlize(data: array) -> array:
    if len(data.shape) != 3:
        raise ValueError(f'data input should be: C,H,W, but got shape:{data.shape}')
    return ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255)

def pair_normlize(data1: array, data2: array) -> array:
    if len(data1.shape) != 3 or len(data2.shape) !=3:
        raise ValueError(f'data input should be: C,H,W, but got shape:{data1.shape} and {data2.shape}')
    up_bound = max(np.max(data1), np.max(data2))
    low_bound = min(np.min(data1), np.min(data2))
    return (data1 - low_bound) / (up_bound - low_bound + 1e-10) * 255, (data2 - low_bound) / (up_bound - low_bound) * 255
