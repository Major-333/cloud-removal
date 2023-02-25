import torch
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP

'''
These code is modified based on:
# https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
'''
class PerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        self.block = torchvision.models.vgg16(pretrained=True).features[:8]
        for p in self.block.parameters():
            p.requires_grad = False
        self.block.eval()
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda())
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        x = self.block(x)
        y = self.block(y)
        loss += torch.nn.functional.l1_loss(x, y)
        return loss
