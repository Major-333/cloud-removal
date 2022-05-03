import numpy as np
from numpy import array
import torch
from torch import tensor

def get_sam(img_true: tensor, img_predict: tensor) -> tensor:
    mat = torch.mul(img_true, img_predict)
    mat = torch.sum(mat, 1)
    mat = torch.div(mat, torch.sqrt(torch.sum(torch.mul(img_true, img_true), 1)))
    mat = torch.div(mat, torch.sqrt(torch.sum(torch.mul(img_predict, img_predict), 1)))
    mat = torch.acos(torch.clamp(mat, -1, 1))
    return torch.mean(mat)
