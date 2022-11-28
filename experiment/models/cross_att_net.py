import torch
import numpy as np
import os
import pdb
import torchvision
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


# base conv block
class BaseConvBlock(nn.Module):
    def __init__(self, inchannel, outchannel) -> None:
        super(BaseConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        return self.block(x)


# resnet模块
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, scale=0.1):
        super(ResidualBlock, self).__init__()
        self.scale = scale
        self.block = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.scale * self.block(x)


# attention block
class AttentionBlock(nn.Module):
    def __init__(self, inchannel, outchannel) -> None:
        super(AttentionBlock, self).__init__()
        self.spatial_block = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1, groups=inchannel),
            nn.PReLU(),
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1, groups=inchannel),
            nn.PReLU(),
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1, groups=inchannel)
        )
        self.channel_block = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(outchannel*2, outchannel, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f_s = self.spatial_block(x)
        f_c = self.channel_block(x)
        concat = torch.cat((f_s, f_c), 1)
        return x * self.conv(concat)


# feature extractor block
class FeatureExtractor(nn.Module):
    def __init__(self, inchannel, outchannel, layernum) -> None:
        super(FeatureExtractor, self).__init__()
        self.block = nn.Sequential(
            ResidualBlock(inchannel, outchannel),
            *[ResidualBlock(outchannel, outchannel) for _ in range(layernum-1)]
        )
    
    def forward(self, x):
        return self.block(x)


# cross attention block
class CrossAttentionBlock(nn.Module):
    def __init__(self, inchannel, outchannel) -> None:
        super(CrossAttentionBlock, self).__init__()
        self.block = nn.Sequential(
            ResidualBlock(inchannel, outchannel),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ResidualBlock(outchannel, outchannel),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ResidualBlock(outchannel, outchannel),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x):
        # pdb.set_trace()
        return self.block(x)


# cross attention cloud removal network
class CrossAttNet(nn.Module):
    def __init__(self, opt_channel, sar_channel):
        super(CrossAttNet, self).__init__()
        self.input_opt = BaseConvBlock(opt_channel, 128)
        self.input_sar = BaseConvBlock(sar_channel, 128)
        self.query_opt = FeatureExtractor(128, 128, 4)
        self.query_sar = FeatureExtractor(128, 128, 3)
        self.attention_opt = CrossAttentionBlock(128, 128)
        self.feature_fusion_block = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            *[nn.Sequential(
                ResidualBlock(256, 256),
                AttentionBlock(256, 256)
            ) for _ in range(3)],
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, opt_channel, kernel_size=3, padding=1)
        )

    def forward(self, input):
        # pdb.set_trace()
        sar = input[:, :2, :, :]
        opt = input[:, 2:, :, :]
        f_opt = self.input_opt(opt)
        f_sar = self.input_sar(sar)
        f_q_opt = self.query_opt(f_opt)
        f_q_sar = self.query_sar(f_sar)
        f_k_opt = self.attention_opt(f_opt)
        f1 = f_q_opt * f_k_opt
        f2 = f_q_opt * f_q_sar
        concat = torch.cat((f1, f2), 1)
        return self.feature_fusion_block(concat)


# ablation networks for cross attention
class AblationNet1(nn.Module):
    def __init__(self, opt_channel, sar_channel):
        super(AblationNet1, self).__init__()
        self.input_opt = BaseConvBlock(opt_channel, 128)
        self.input_sar = BaseConvBlock(sar_channel, 128)
        self.query_opt = FeatureExtractor(128, 128, 4)
        self.query_sar = FeatureExtractor(128, 128, 3)
        self.attention_sar = CrossAttentionBlock(128, 128)
        self.feature_fusion_block = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            *[nn.Sequential(
                ResidualBlock(256, 256),
                AttentionBlock(256, 256)
            ) for _ in range(3)],
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, opt_channel, kernel_size=3, padding=1)
        )

    def forward(self, input):
        # pdb.set_trace()
        sar = input[:, :2, :, :]
        opt = input[:, 2:, :, :]
        f_opt = self.input_opt(opt)
        f_sar = self.input_sar(sar)
        f_q_opt = self.query_opt(f_opt)
        f_q_sar = self.query_sar(f_sar)
        f_k_sar = self.attention_sar(f_sar)
        f1 = f_q_opt * f_k_sar
        f2 = f_q_opt * f_q_sar
        concat = torch.cat((f1, f2), 1)
        return self.feature_fusion_block(concat)


class AblationNet2(nn.Module):
    def __init__(self, opt_channel, sar_channel):
        super(AblationNet2, self).__init__()
        self.input_opt = BaseConvBlock(opt_channel, 128)
        self.input_sar = BaseConvBlock(sar_channel, 128)
        self.query_opt = FeatureExtractor(128, 128, 4)
        self.query_sar = FeatureExtractor(128, 128, 3)
        self.feature_fusion_block = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            *[nn.Sequential(
                ResidualBlock(256, 256),
                AttentionBlock(256, 256)
            ) for _ in range(3)],
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, opt_channel, kernel_size=3, padding=1)
        )

    def forward(self, input):
        # pdb.set_trace()
        sar = input[:, :2, :, :]
        opt = input[:, 2:, :, :]
        f_opt = self.input_opt(opt)
        f_sar = self.input_sar(sar)
        f_q_opt = self.query_opt(f_opt)
        f_q_sar = self.query_sar(f_sar)
        concat = torch.cat((f_q_opt, f_q_sar), 1)
        return self.feature_fusion_block(concat)


class AblationNet3(nn.Module):
    def __init__(self, opt_channel, sar_channel):
        super(AblationNet3, self).__init__()
        self.input_opt = BaseConvBlock(opt_channel, 128)
        self.input_sar = BaseConvBlock(sar_channel, 128)
        self.query_opt = FeatureExtractor(128, 128, 4)
        self.query_sar = FeatureExtractor(128, 128, 3)
        self.attention_sar = CrossAttentionBlock(128, 128)
        self.feature_fusion_block = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            *[nn.Sequential(
                ResidualBlock(256, 256),
                AttentionBlock(256, 256)
            ) for _ in range(3)],
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, opt_channel, kernel_size=3, padding=1)
        )

    def forward(self, input):
        # pdb.set_trace()
        sar = input[:, :2, :, :]
        opt = input[:, 2:, :, :]
        f_opt = self.input_opt(opt)
        f_sar = self.input_sar(sar)
        f_q_opt = self.query_opt(f_opt)
        f_q_sar = self.query_sar(f_sar)
        f_k_opt = self.attention_sar(f_opt)
        f1 = f_q_sar * f_k_opt
        f2 = f_q_opt * f_q_sar
        concat = torch.cat((f1, f2), 1)
        return self.feature_fusion_block(concat)


if __name__ == '__main__':
    test_opt = torch.randn(16, 13, 64, 64)
    test_sar = torch.randn(16, 2, 64, 64)
    input = torch.cat((test_sar, test_opt), 1)
    # model = CrossAttNet(13, 2)
    model = AblationNet3(13, 2)
    output = model(input)
    print(output.size())
    assert test_opt.size() == output.size()