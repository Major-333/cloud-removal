import torch
from loss.charbonnier_loss import CharbonnierLoss
from loss.simulation_fusion_gan_loss import SimulationFusionGANLoss
from torch.nn.parallel import DistributedDataParallel as DDP

LOSS_MAPPER = {
    'L1Loss': torch.nn.L1Loss(),
    'MSE': torch.nn.MSELoss(),
    'CharbonnierLoss': CharbonnierLoss(),
    'SimulationFusionGANLoss': SimulationFusionGANLoss()
}


def build_loss_fn(loss_name: str):
    return LOSS_MAPPER[loss_name]

def build_cuda_loss_fn(loss_name: str, gpu_id: int):
    loss_fn = LOSS_MAPPER[loss_name]
    loss_fn = loss_fn.cuda()
    return loss_fn