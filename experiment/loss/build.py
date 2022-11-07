import torch
from loss.charbonnier_loss import CharbonnierLoss
from loss.simulation_fusion_gan_loss import SimulationFusionGANLoss

LOSS_MAPPER = {
    'MSE': torch.nn.MSELoss(),
    'SE': torch.nn.MSELoss(reduction='sum'),
    'CharbonnierLoss': CharbonnierLoss(),
    'L1Loss': torch.nn.L1Loss(),
    # 'SimulationFusionGANLoss': SimulationFusionGANLoss()
}


def build_loss_fn(loss_name: str):
    return LOSS_MAPPER[loss_name]