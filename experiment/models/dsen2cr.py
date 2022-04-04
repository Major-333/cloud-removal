import torch
import numpy as np
import os
import torchvision
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

# resnet模块


class ResidualBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1, scale=0.1):
        super(ResidualBlock, self).__init__()
        self.scale = scale
        self.block = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
                                   nn.BatchNorm2d(outchannel), nn.ReLU(inplace=True),
                                   nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(outchannel))
        self.shortcut = nn.Sequential()

    def forward(self, x):
        return x + self.scale * self.block(x)


class DSen2_CR(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers=6, feature_dim=256):
        super(DSen2_CR, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=feature_dim, kernel_size=3, bias=True, stride=1, padding=1),
            nn.ReLU(True), *[ResidualBlock(feature_dim, feature_dim) for i in range(num_layers)],
            nn.Conv2d(feature_dim, out_channels, kernel_size=3, bias=False, stride=1, padding=1))

    def forward(self, x):
        return x[:, 2:, :, :] + self.blocks(x)


if __name__ == '__main__':
    fake_input = torch.rand(4, 15, 256, 256)
    print(fake_input)
    net = DSen2_CR(15, 13)
    output = net(fake_input)
    print(output.shape)
