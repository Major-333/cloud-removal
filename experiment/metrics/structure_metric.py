import numpy as np
from numpy import array
import torch
from torch import Tensor

def get_sam(img_true: Tensor, img_predict: Tensor) -> Tensor:
    mat = torch.mul(img_true, img_predict)
    mat = torch.sum(mat, 1)
    mat = torch.div(mat, torch.sqrt(torch.sum(torch.mul(img_true, img_true), 1)) + 1e-6)
    mat = torch.div(mat, torch.sqrt(torch.sum(torch.mul(img_predict, img_predict), 1)) + 1e-6)
    mat = torch.acos(torch.clamp(mat, -1, 1))
    return torch.mean(mat)
