import torch
import torch.nn as nn

import torch
import torch.nn as nn
from utils import convert_range

'''
Simulation net and its building blocks:
- FusionNet
- FusionInConv
- SimDown
- SimMiddleConv
- SimUp
- SimOutConv
- MyColorJitter
'''
class SimulationNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SimulationNet, self).__init__()
        self.in_conv = SimInConv(in_channels, 64)
        self.down1 = SimDown(64, 128)
        self.down2 = SimDown(128, 256)
        self.down3 = SimDown(256, 512)
        self.down4 = SimDown(512, 512)
        self.down5 = SimDown(512, 512)
        self.middle_conv = SimMiddleConv(512, 512)
        self.middle_up = SimUp(512, 512)
        self.up5 = SimUp(1024, 512)
        self.up4 = SimUp(1024, 512)
        self.up3 = SimUp(1024, 256)
        self.up2 = SimUp(512, 128)
        self.up1 = SimUp(256, 64)
        self.out_conv = SimOutConv(128, out_channels)
        self.color_jitter = MyColorJitter()

    def forward(self, concated_corrupted_sar_image):
        concated_corrupted_sar_image = original_to_input(concated_corrupted_sar_image)
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
        simulated_image = out_conv_feature_map
        simulated_image = output_to_original(simulated_image)
        '''
        If model is not training
        Then according to paper
        We randomly alter contrast and luminance of the simulated optical image
        '''
        if not self.training:
            simulated_image = self.color_jitter(simulated_image)
        return simulated_image


class SimInConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SimInConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.leakey_relu = nn.LeakyReLU(0.2, True)

    def forward(self, sar_image):
        in_conv_feature_map = self.conv(sar_image)
        in_conv_feature_map = self.leakey_relu(in_conv_feature_map)
        return in_conv_feature_map


class SimDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SimDown, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leakey_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, last_feature_map):
        down_feature_map = self.conv(last_feature_map)
        down_feature_map = self.batch_norm(down_feature_map)
        down_feature_map = self.leakey_relu(down_feature_map)
        return down_feature_map


class SimMiddleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SimMiddleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, last_feature_map):
        middle_conv_feature_map = self.conv(last_feature_map)
        middle_conv_feature_map = self.relu(middle_conv_feature_map)
        return middle_conv_feature_map


class SimUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SimUp, self).__init__()
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


class SimOutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SimOutConv, self).__init__()
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


class MyColorJitter(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, img):
        img = self.adjust_brightness(img)
        img = self.adjust_contrast(img)
        return img

    def adjust_brightness(self, img):
        factor = float(torch.empty(1).uniform_(0.5, 1.5))
        return self.blend(img, torch.zeros_like(img), factor)

    def adjust_contrast(self, img):
        factor = float(torch.empty(1).uniform_(0.5, 1.5))
        mean = torch.mean(img, dim=(-3, -2, -1), keepdim=True)
        return self.blend(img, mean, factor)

    def blend(self, img1, img2, ratio):
        ratio = float(ratio)
        bound = 1.0
        return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)

'''
Fusion net and its building blocks:
- SimulationNet
- SimInConv
- FusionDown
- FusionMiddleConv
- FusionUp
- FusionOutConv
'''
class FusionNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FusionNet, self).__init__()
        self.in_conv = FusionInConv(in_channels, 64)
        self.down1 = FusionDown(64, 128)
        self.down2 = FusionDown(128, 256)
        self.down3 = FusionDown(256, 512)
        self.down4 = FusionDown(512, 512)
        self.down5 = FusionDown(512, 512)
        self.middle_conv = FusionMiddleConv(512, 512)
        self.middle_up = FusionUp(512, 512)
        self.up5 = FusionUp(1024, 512)
        self.up4 = FusionUp(1024, 512)
        self.up3 = FusionUp(1024, 256)
        self.up2 = FusionUp(512, 128)
        self.up1 = FusionUp(256, 64)
        self.out_conv = FusionOutConv(128, out_channels)

    def forward(self, simulation_image, concated_corrupted_sar_image):
        concated_corrupted_sar_image = original_to_input(concated_corrupted_sar_image)
        concated_simulation_corrupted_sar_image = torch.cat((simulation_image, concated_corrupted_sar_image), dim=1)
        in_feature_map = self.in_conv(concated_simulation_corrupted_sar_image)
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
        simulated_image = out_conv_feature_map
        simulated_image = output_to_original(simulated_image)
        return simulated_image


class FusionInConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FusionInConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.leakey_relu = nn.LeakyReLU(0.2, True)

    def forward(self, sar_image):
        in_conv_feature_map = self.conv(sar_image)
        in_conv_feature_map = self.leakey_relu(in_conv_feature_map)
        return in_conv_feature_map


class FusionDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FusionDown, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.leakey_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, last_feature_map):
        down_feature_map = self.conv(last_feature_map)
        down_feature_map = self.leakey_relu(down_feature_map)
        return down_feature_map


class FusionMiddleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FusionMiddleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, last_feature_map):
        middle_conv_feature_map = self.conv(last_feature_map)
        middle_conv_feature_map = self.relu(middle_conv_feature_map)
        return middle_conv_feature_map


class FusionUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FusionUp, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels,
                                                 out_channels,
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1,
                                                 bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, last_feature_map, skip_feature_map):
        up_feature_map = self.conv_transpose(last_feature_map)
        up_feature_map = self.relu(up_feature_map)
        concated_feature_map = torch.cat([skip_feature_map, up_feature_map], dim=1)
        return concated_feature_map


class FusionOutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FusionOutConv, self).__init__()
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

'''
Discriminative net and its building blocks:
- DiscriminativeNet
- DiscDown
- DiscMiddleConv
- DiscOutConv
'''
class DiscriminativeNet(nn.Module):

    def __init__(self, in_channels):
        super(DiscriminativeNet, self).__init__()
        self.down1 = DiscDown(in_channels, 64)
        self.down2 = DiscDown(64, 128)
        self.middle_conv = DiscMiddleConv(128, 256)
        self.out_conv = DiscOutConv(256)

    def forward(self, concated_optical_sar_image):
        concated_optical_sar_image = original_to_input(concated_optical_sar_image)
        down_feature_map_1 = self.down1(concated_optical_sar_image)
        down_feature_map_2 = self.down2(down_feature_map_1)
        middle_conv_feature_map = self.middle_conv(down_feature_map_2)
        prediction_map = self.out_conv(middle_conv_feature_map)
        return prediction_map


class DiscDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DiscDown, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.leakey_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, last_feature_map):
        down_feature_map = self.conv(last_feature_map)
        down_feature_map = self.batch_norm(down_feature_map)
        down_feature_map = self.leakey_relu(down_feature_map)
        return down_feature_map


class DiscMiddleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DiscMiddleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=1, bias=False)
        self.leakey_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, last_feature_map):
        middle_conv_feature_map = self.conv(last_feature_map)
        middle_conv_feature_map = self.leakey_relu(middle_conv_feature_map)
        return middle_conv_feature_map


class DiscOutConv(nn.Module):

    def __init__(self, in_channels):
        super(DiscOutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, last_feature_map):
        out_conv_feature_map = self.conv(last_feature_map)
        prediction_map = self.sigmoid(out_conv_feature_map)
        return prediction_map

def original_to_input(original):
    return convert_range(original, (0,1), (-1,1))

def output_to_original(output):
    return convert_range(output, (-1,1), (0,1))