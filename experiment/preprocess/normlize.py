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


def normlize_to_uint8_foreach_channel(data: array) -> array:
    if len(data.shape) != 4:
        raise ValueError(f'data input should be: N,C,H,W, but got shape:{data.shape}')
    total_channels = data.shape[1]
    for index in range(total_channels):
        data_slice = data[:, index, :, :]
        data[:, index, :, :] = (data_slice - np.min(data_slice)) / \
            (np.max(data_slice) - np.min(data_slice) + 1e-10) * 255
    return data.astype('uint8')


def normlize_to_uint8(data: array) -> array:
    return ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype('uint8')

