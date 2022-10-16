import torch.nn as nn


class DiscriminativeNet(nn.Module):
    def __init__(self, in_channels):
        super(DiscriminativeNet, self).__init__()
        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.middle_conv = MiddleConv(128, 256)
        self.out_conv = OutConv(256)

    def forward(self, concated_optical_sar_image):
        down_feature_map_1 = self.down1(concated_optical_sar_image)
        down_feature_map_2 = self.down2(down_feature_map_1)
        middle_conv_feature_map = self.middle_conv(down_feature_map_2)
        prediction_map = self.out_conv(middle_conv_feature_map)
        return prediction_map


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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=1, bias=False)
        self.leakey_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, last_feature_map):
        middle_conv_feature_map = self.conv(last_feature_map)
        middle_conv_feature_map = self.leakey_relu(middle_conv_feature_map)
        return middle_conv_feature_map


class OutConv(nn.Module):

    def __init__(self, in_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, last_feature_map):
        out_conv_feature_map = self.conv(last_feature_map)
        prediction_map = self.sigmoid(out_conv_feature_map)
        return prediction_map