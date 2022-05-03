import os
import sys
import math
import torch
from torch import tensor
import numpy as np
import cv2


def get_psnr(img1: tensor, img2: tensor, maxi: float = 255.0): 
    rmse = get_rmse(img1, img2)
    return 20 * torch.log10(maxi / rmse)


def get_rmse(img1: tensor, img2: tensor):
    mse = torch.mean((img1 - img2)**2)
    return torch.sqrt(mse)

def get_mae(img1: tensor, img2: tensor):
    mae = torch.mean(torch.abs(img1 - img2))
    return mae
