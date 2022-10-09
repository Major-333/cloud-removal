import torch
from loss.charbonnier_loss import CharbonnierLoss

LOSS_MAPPER = {'MSE': torch.nn.MSELoss(), 'CharbonnierLoss': CharbonnierLoss()}


def build_loss_fn(loss_name: str):
    return LOSS_MAPPER[loss_name]