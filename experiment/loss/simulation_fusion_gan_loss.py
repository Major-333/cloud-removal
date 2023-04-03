import torch
import torch.nn as nn
from loss.perceptual_loss import PerceptualLoss


class SimulationFusionGANLoss(nn.Module):

    def __init__(self) -> None:
        super(SimulationFusionGANLoss, self).__init__()
        self.weight_L1 = 10
        self.weight_local = 10
        self.weight_GAN = 1
        self.weight_perceptual = 0.8

        self.loss_fn_L1 = nn.L1Loss()
        self.loss_fn_local = nn.L1Loss()
        self.loss_fn_GAN = nn.MSELoss()
        self.loss_fn_perceptual = PerceptualLoss()

    def forward(self, pred_fake, fused_image, ground_truth, cloud_mask):
        loss_L1 = self.loss_fn_L1(fused_image, ground_truth) / 2
        loss_local = self.loss_fn_local(
            torch.masked_fill(input=fused_image, mask=cloud_mask.unsqueeze(1) == 0, value=0),
            torch.masked_fill(input=ground_truth, mask=cloud_mask.unsqueeze(1) == 0, value=0)) / 2
        loss_GAN = self.loss_fn_GAN(pred_fake, torch.ones(pred_fake.shape).cuda())
        loss_perceptual = self.loss_fn_perceptual(torch.flip((fused_image[:, 1:4, :, :] + 1) / 2, dims=[1]),
                                                  torch.flip((ground_truth[:, 1:4, :, :] + 1) / 2, dims=[1]))
        return self.weight_L1 * loss_L1 + self.weight_local * loss_local + self.weight_GAN * loss_GAN + self.weight_perceptual * loss_perceptual, loss_L1, loss_local, loss_GAN, loss_perceptual