import os
import logging
from typing import List, Tuple
import pandas as pd
import numpy as np
from numpy import array
from dataset.basic_dataloader import SEN12MSCRDataset, Seasons, S1Bands, S2Bands
from dataset.processed_dataloader import get_s1s2s2cloudy_processed_triplet
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis, figure
import seaborn as sns


def get_output_with_groundtruth_distribution_by_channel(ground_truth: array,
                                                        output: array,
                                                        channel_idx: int,
                                                        filepath: str = None) -> figure:
    logging.info(f'will get the distribution of channel:{channel_idx}')
    # FIXME: reset the plt
    plt.clf()
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.reset_defaults()
    if len(ground_truth.shape) == 4 or len(output.shape) == 4:
        groundtruth_channel = ground_truth[:, channel_idx, :, :].reshape(-1)
        output_channel = output[:, channel_idx, :, :].reshape(-1)
    elif len(ground_truth.shape) == 3 or len(output.shape) == 3:
        groundtruth_channel = ground_truth[channel_idx, :, :].reshape(-1)
        output_channel = output[channel_idx, :, :].reshape(-1)
    else:
        err_msg = f'Shape Error! getting the distribution, but graound truth shape: {ground_truth.shape}, output shape:{output.shape}'
        logging.error(err_msg)
        raise ValueError(err_msg)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.setp(axes, xticks=[], yticks=[])
    df1 = pd.DataFrame({
        f'groundtruth_channel:{channel_idx}': groundtruth_channel,
    })
    fig.add_subplot(1, 2, 1)
    sns.histplot(df1)
    df2 = pd.DataFrame({
        f'output_channel:{channel_idx}': output_channel,
    })
    fig.add_subplot(1, 2, 2)
    sns.histplot(df2)
    if filepath:
        plt.savefig(filepath)
    fig.tight_layout()
    return figure


def visualize_output_with_groundtruth_only_rgb(ground_truth: array, output: array, input: array, filepath: str,
                                               title: str) -> figure:
    try:
        logging.info('will visualize rgb comparsion')
        plt.clf()
        mpl.rcParams.update(mpl.rcParamsDefault)
        if len(ground_truth.shape) == 4 and len(output.shape) == 4:
            sample = np.random.choice(output.shape[0], 1)[0]
            ground_truth = ground_truth[sample, :, :, :]
            output = output[sample, :, :, :]
            input = input[sample, :, :, :]
        sar = input[:2, :, :].copy()
        ms = input[2:, :, :].copy()
        if len(ground_truth.shape) != 3 or len(output.shape) != 3:
            err_msg = f'Shape Error! visualizing, but graound truth shape: {ground_truth.shape}, output shape:{output.shape}'
            logging.error(err_msg)
            raise ValueError(err_msg)
        rows, cols = 2, 2
        bands = S2Bands.ALL.value
        # fig, axes = plt.subplots(rows, cols, figsize=(12, 96))
        fig = plt.figure()
        fig.set_size_inches(12, 12)
        axes = [fig.add_subplot(rows, cols, r * cols + c + 1) for r in range(0, rows) for c in range(0, cols)]
        images = [ground_truth, output, ms, sar]
        logging.info(f'images shape:{ground_truth.shape}, {output.shape}, {ms.shape}, {sar.shape}')
        titles = ['RGB_ground_truth', 'RGB_output', 'RGB_input', 'SAR_input']
        for index, ax in enumerate(axes):
            channel_idxs = list(map(lambda x: x - 1, S2Bands.RGB.value))
            print(images[index].shape)
            if index == 3:
                sar_vh = images[index][0, :, :]
                sar_vv = images[index][1, :, :]
                img = np.dstack((sar_vh, sar_vv, np.zeros(sar_vv.shape)))
                img = np.transpose(img, (1, 0, 2))
            else:
                img = images[index][channel_idxs, :, :]
                img = np.transpose(img, (2, 1, 0))
            img = (img * 255.0 / (np.max(img) + 1e-7))
            img = img.astype('int')
            ax.set_title(titles[index])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.subplots_adjust(top=0.92)
        fig.suptitle(title, fontsize=20)
        logging.info(f'will save the visualization media to:{filepath}')
        plt.savefig(filepath)
    except Exception as e:
        logging.error(f'failed to visualize:{str(e)}')
        raise ValueError(str(e))
    return fig


def visualize_output_with_groundtruth(ground_truth: array, output: array, filepath: str, title: str) -> figure:
    plt.clf()
    mpl.rcParams.update(mpl.rcParamsDefault)
    rows, cols = 14, 2
    bands = S2Bands.ALL.value
    # fig, axes = plt.subplots(rows, cols, figsize=(12, 96))
    fig = plt.figure()
    fig.set_size_inches(12, 96)
    axes = [fig.add_subplot(rows, cols, r * cols + c + 1) for r in range(0, rows) for c in range(0, cols)]
    for index, ax in enumerate(axes):
        row_idx = index // 2
        col_idx = index % 2
        if row_idx == 0:
            channel_idxs = list(map(lambda x: x - 1, S2Bands.RGB.value))
            if col_idx == 0:
                img = ground_truth[channel_idxs, :, :]
            else:
                img = output[channel_idxs, :, :]
            img = np.transpose(img, (2, 1, 0))
            img = img * 255.0 / np.max(img)
            img = img.astype('int')
            if col_idx == 0:
                ax.set_title('RGB_ground_truth')
            else:
                ax.set_title('RGB_output')
            ax.imshow(img)
        else:
            band = S2Bands(bands[row_idx - 1])
            channel_idx = band.value - 1
            if col_idx == 0:
                img = ground_truth[channel_idx, :, :]
            else:
                img = output[channel_idx, :, :]
            img = np.transpose(img, (1, 0))
            img = img * 255.0 / np.max(img)
            img = img.astype('int')
            if col_idx == 0:
                ax.set_title(f'Band:{band.name}_ground_truth')
            else:
                ax.set_title(f'Band:{band.name}_output')
            # plt.imshow(img, cmap='gray')
            ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(top=0.98)
    fig.suptitle(title, fontsize=20)
    plt.savefig(filepath)
    return fig


if __name__ == '__main__':
    base_dir = '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data/SEN12MS_CR'
    processed_dir = '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data/SEN12MS_CR_PROCESSED_V2'
    dataloader = SEN12MSCRDataset(base_dir)
    scene_id = 102
    # s1, s2, s2cloudy, _ = dataloader.get_triplets(Seasons.SUMMER,
    #                                               scene_ids=scene_id,
    #                                               s1_bands=S1Bands.ALL,
    #                                               s2_bands=S2Bands.ALL,
    #                                               s2cloudy_bands=S2Bands.ALL)
    # get_output_with_groundtruth_distribution_by_channel(
    #     s2, s2cloudy, 1, filepath='./c.png')

    # visualize_output_with_groundtruth(
    #     s2[0, :, :, :], s2cloudy[0, :, :, :], './a.png', title='test-visualization')
    # visualize_output_with_groundtruth_only_rgb(
    #     s2[0, :, :, :], s2cloudy[0, :, :, :], './a1.png', title='test-visualization')
    scene_ids_dict = dataloader.get_season_ids(Seasons.SUMMER)
    patch_id = scene_ids_dict[scene_id][115]
    print(patch_id)
    season = Seasons.SUMMER
    s1, s2, s2cloudy = get_s1s2s2cloudy_processed_triplet(processed_dir, season, scene_id, patch_id)
    print(np.min(s1), np.max(s1))
    print(np.min(s2), np.max(s2))
    print(np.min(s2cloudy), np.max(s2cloudy))
    input = np.concatenate((s1, s2cloudy), axis=0)
    s1 = np.expand_dims(s1, axis=0)
    s2 = np.expand_dims(s2, axis=0)
    s2cloudy = np.expand_dims(s2cloudy, axis=0)
    input = np.expand_dims(input, axis=0)

    # get_output_with_groundtruth_distribution_by_channel(
    #     s2, s2cloudy, 1, filepath='./d2.png')
    # visualize_output_with_groundtruth(
    #     s2, s2cloudy, './b2.png', title='test-visualization')
    visualize_output_with_groundtruth_only_rgb(s2, s2cloudy, input, './g.png', title='test-visualization')
    # get_output_with_groundtruth_distribution_by_channel(
    #     s2, s2cloudy, 1, filepath='./e2.png')
