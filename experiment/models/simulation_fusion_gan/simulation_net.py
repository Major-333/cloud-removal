import torch
import torch.nn as nn


class SimulationNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SimulationNet, self).__init__()
        self.in_conv = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.down5 = Down(512, 512)
        self.middle_conv = MiddleConv(512, 512)
        self.middle_up = Up(512, 512)
        self.up5 = Up(1024, 512)
        self.up4 = Up(1024, 512)
        self.up3 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up1 = Up(256, 64)
        self.out_conv = OutConv(128, out_channels)

    def forward(self, concated_corrupted_sar_image):
        sar_image = concated_corrupted_sar_image[:, :2, :, :]
        in_feature_map = self.in_conv(sar_image)
        down_feature_map_1 = self.down1(in_feature_map)
        down_feature_map_2 = self.down2(down_feature_map_1)
        down_feature_map_3 = self.down3(down_feature_map_2)
        down_feature_map_4 = self.down4(down_feature_map_3)
        down_feature_map_5 = self.down5(down_feature_map_4)
        middle_conv_feature_map = self.middle_conv(down_feature_map_5)
        middle_up_feature_map = self.middle_up(middle_conv_feature_map, down_feature_map_5)
        up_feature_map_5 = self.up5(middle_up_feature_map, down_feature_map_4)
        up_feature_map_4 = self.up4(up_feature_map_5, down_feature_map_3)
        up_feature_map_3 = self.up3(up_feature_map_4, down_feature_map_2)
        up_feature_map_2 = self.up2(up_feature_map_3, down_feature_map_1)
        up_feature_map_1 = self.up1(up_feature_map_2, in_feature_map)
        out_conv_feature_map = self.out_conv(up_feature_map_1)
        simulated_image = (out_conv_feature_map + 1) / 2 * 255
        return simulated_image


class InConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.leakey_relu = nn.LeakyReLU(0.2, True)

    def forward(self, sar_image):
        in_conv_feature_map = self.conv(sar_image)
        in_conv_feature_map = self.leakey_relu(in_conv_feature_map)
        return in_conv_feature_map


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leakey_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, last_feature_map):
        down_feature_map = self.conv(last_feature_map)
        down_feature_map = self.batch_norm(down_feature_map)
        down_feature_map = self.leakey_relu(down_feature_map)
        return down_feature_map


class MiddleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MiddleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, last_feature_map):
        middle_conv_feature_map = self.conv(last_feature_map)
        middle_conv_feature_map = self.relu(middle_conv_feature_map)
        return middle_conv_feature_map


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels,
                                                 out_channels,
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1,
                                                 bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, last_feature_map, skip_feature_map):
        up_feature_map = self.conv_transpose(last_feature_map)
        up_feature_map = self.batch_norm(up_feature_map)
        up_feature_map = self.relu(up_feature_map)
        concated_feature_map = torch.cat([skip_feature_map, up_feature_map], dim=1)
        return concated_feature_map


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels,
                                                 out_channels,
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1,
                                                 bias=False)
        self.tanh = nn.Tanh()

    def forward(self, last_feature_map):
        out_conv_feature_map = self.conv_transpose(last_feature_map)
        out_conv_feature_map = self.tanh(out_conv_feature_map)
        return out_conv_feature_map