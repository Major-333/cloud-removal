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

def build_distributed_loss_fn(loss_name: str, gpu_id: int):
    loss_fn = LOSS_MAPPER[loss_name](gpu_id)
    return DDP(loss_fn, device_ids=[gpu_id])

def build_distributed_gan_loss_fn(loss_name: str, gpu_id: int):
    loss_fn = LOSS_MAPPER[loss_name](gpu_id)
    return DDP(loss_fn, device_ids=[gpu_id], broadcast_buffers=False)