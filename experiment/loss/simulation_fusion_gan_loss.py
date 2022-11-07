import torch
import torch.nn as nn
from loss.perceptual_loss import PerceptualLoss


class SimulationFusionGANLoss(nn.Module):

    def __init__(self, local_rank) -> None:
        super(SimulationFusionGANLoss, self).__init__()
        self.weight_L1 = 10
        self.weight_local = 10
        self.weight_GAN = 1
        self.weight_perceptual = 0.8
        self.loss_fn_L1 = nn.L1Loss()
        self.loss_fn_GAN = nn.MSELoss()
        self.loss_fn_perceptual = PerceptualLoss(local_rank=local_rank)

    def forward(self, pred_fake, fused_image, ground_truth):
        loss_L1 = self.loss_fn_L1(fused_image, ground_truth) / 2
        loss_GAN = self.loss_fn_GAN(pred_fake, torch.ones(pred_fake.shape).cuda())
        loss_perceptual = self.loss_fn_perceptual(torch.flip((fused_image[:, 1:4, :, :] + 1) / 2, dims=[1]),
                                                  torch.flip((ground_truth[:, 1:4, :, :] + 1) / 2, dims=[1]))
        return self.weight_L1 * loss_L1 + self.weight_GAN * loss_GAN + self.weight_perceptual * loss_perceptual, loss_L1, loss_GAN, loss_perceptual