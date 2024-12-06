import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class Aggregation(nn.Module):
    def __init__(self, channel):
        super(Aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class NestedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32, pretrained=True):
        super(NestedUNet, self).__init__()
        self.base_channels = base_channels

        # Encoder (Using ResNet-34 as backbone)
        resnet = models.resnet34(pretrained=pretrained)

        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder1 = resnet.layer1  # 64 channels
        self.encoder2 = resnet.layer2  # 128 channels
        self.encoder3 = resnet.layer3  # 256 channels
        self.encoder4 = resnet.layer4  # 512 channels

        # RFB modules
        self.rfb1 = RFB_modified(64, base_channels)
        self.rfb2 = RFB_modified(128, base_channels)
        self.rfb3 = RFB_modified(256, base_channels)
        self.rfb4 = RFB_modified(512, base_channels)

        # Aggregation module
        self.agg = Aggregation(base_channels)

        # Decoder
        self.decoder00 = ConvBlock(base_channels, base_channels)
        self.decoder10 = ConvBlock(base_channels, base_channels)
        self.decoder20 = ConvBlock(base_channels, base_channels)
        self.decoder30 = ConvBlock(base_channels, base_channels)

        self.decoder01 = ConvBlock(base_channels * 2, base_channels)
        self.decoder11 = ConvBlock(base_channels * 2, base_channels)
        self.decoder21 = ConvBlock(base_channels * 2, base_channels)

        self.decoder02 = ConvBlock(base_channels * 3, base_channels)
        self.decoder12 = ConvBlock(base_channels * 3, base_channels)

        self.decoder03 = ConvBlock(base_channels * 4, base_channels)

        # Final Convolution
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        x0 = self.encoder0(x)  # [B, 64, H/4, W/4]
        x1 = self.encoder1(x0)  # [B, 64, H/4, W/4]
        x2 = self.encoder2(x1)  # [B, 128, H/8, W/8]
        x3 = self.encoder3(x2)  # [B, 256, H/16, W/16]
        x4 = self.encoder4(x3)  # [B, 512, H/32, W/32]

        # RFB modules
        rfb1 = self.rfb1(x1)
        rfb2 = self.rfb2(x2)
        rfb3 = self.rfb3(x3)
        rfb4 = self.rfb4(x4)

        # Attention map
        agg_map = self.agg(rfb4, rfb3, rfb2)
        agg_map_up = F.interpolate(agg_map, scale_factor=8, mode='bilinear', align_corners=True)

        # Decoder path (Nested)
        d00 = self.decoder00(rfb1)
        d10 = self.decoder10(rfb2)
        d20 = self.decoder20(rfb3)
        d30 = self.decoder30(rfb4)

        d01 = self.decoder01(torch.cat([self.up(d10), d00], dim=1))
        d11 = self.decoder11(torch.cat([self.up(d20), d10], dim=1))
        d21 = self.decoder21(torch.cat([self.up(d30), d20], dim=1))

        d02 = self.decoder02(torch.cat([self.up(d11), d01, d00], dim=1))
        d12 = self.decoder12(torch.cat([self.up(d21), d11, d10], dim=1))

        d03 = self.decoder03(torch.cat([self.up(d12), d02, d01, d00], dim=1))

        # Final output
        out = self.final_conv(d03)
        # Ensure the output size matches the input size
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)

        return out, agg_map_up


if __name__ == '__main__':
    model = NestedUNet(in_channels=3, out_channels=1, base_channels=32, pretrained=True).cuda()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    output, attention = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Attention map shape: {attention.shape}")