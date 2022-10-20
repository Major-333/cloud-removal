import torch
import torch.nn as nn
from perceptual_loss import PerceptualLoss


class SimulationFusionGANLoss(nn.Module):

    def __init__(self) -> None:
        super(SimulationFusionGANLoss, self).__init__()
        self.weight_L1 = 10
        self.weight_GAN = 1e-4
        self.weight_perceptual = 0.8
        self.loss_fn_L1 = nn.L1Loss()
        self.loss_fn_GAN = nn.MSELoss()
        self.loss_fn_perceptual = PerceptualLoss()

    def forward(self, pred_fake, fused_image, ground_truth):
        loss_L1 = self.loss_fn_L1(fused_image, ground_truth) / 255
        loss_GAN = self.loss_fn_GAN(pred_fake, torch.ones(pred_fake.shape).cuda())
        loss_perceptual = self.loss_fn_perceptual(fused_image / 255,
                                                  ground_truth / 255,
                                                  feature_layers=[1],
                                                  style_layers=[])
        return self.weight_L1 * loss_L1 + self.weight_GAN * loss_GAN + self.weight_perceptual * loss_perceptual, loss_L1, loss_GAN, loss_perceptual