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
    # Dense aggregation module for feature fusion
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


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64, pretrained=True):
        super(UNet, self).__init__()

        # ---- Encoder (Using ResNet Backbone) ----
        self.encoder = models.resnet50(pretrained=pretrained)

        self.encoder_layers = list(self.encoder.children())
        self.layer0 = nn.Sequential(
            self.encoder_layers[0],  # conv1
            self.encoder_layers[1],  # bn1
            self.encoder_layers[2],  # relu
            self.encoder_layers[3]  # maxpool
        )
        self.layer1 = self.encoder_layers[4]  # layer1
        self.layer2 = self.encoder_layers[5]  # layer2
        self.layer3 = self.encoder_layers[6]  # layer3
        self.layer4 = self.encoder_layers[7]  # layer4

        # ---- Receptive Field Blocks for Encoder Features ----
        self.rfb1 = RFB_modified(256, base_channels)
        self.rfb2 = RFB_modified(512, base_channels)
        self.rfb3 = RFB_modified(1024, base_channels)
        self.rfb4 = RFB_modified(2048, base_channels)

        # ---- Aggregation Module ----
        self.agg = Aggregation(base_channels)

        # ---- Decoder Layers ----
        self.up4 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.conv4 = BasicConv2d(2 * base_channels, base_channels, 3, padding=1)

        self.up3 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.conv3 = BasicConv2d(2 * base_channels, base_channels, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.conv2 = BasicConv2d(2 * base_channels, base_channels, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.conv1 = BasicConv2d(2 * base_channels, base_channels, 3, padding=1)

        # ---- Final Convolution ----
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # ---- Encoder ----
        x0 = self.layer0(x)  # [B, 64, H/4, W/4]
        x1 = self.layer1(x0)  # [B, 256, H/4, W/4]
        x2 = self.layer2(x1)  # [B, 512, H/8, W/8]
        x3 = self.layer3(x2)  # [B, 1024, H/16, W/16]
        x4 = self.layer4(x3)  # [B, 2048, H/32, W/32]

        # ---- Apply RFB Modules ----
        rfb1 = self.rfb1(x1)  # [B, base_channels, H/4, W/4]
        rfb2 = self.rfb2(x2)  # [B, base_channels, H/8, W/8]
        rfb3 = self.rfb3(x3)  # [B, base_channels, H/16, W/16]
        rfb4 = self.rfb4(x4)  # [B, base_channels, H/32, W/32]

        # ---- Aggregation to Get Attention Map ----
        agg_map = self.agg(rfb4, rfb3, rfb2)  # [B, 1, H/8, W/8]
        agg_map_up = F.interpolate(agg_map, scale_factor=8, mode='bilinear', align_corners=True)  # [B, 1, H, W]

        # ---- Decoder with Skip Connections ----
        # Decoder Stage 4
        up4 = self.up4(rfb4)  # [B, base_channels, H/16, W/16]
        up4 = torch.cat([up4, rfb3], dim=1)  # [B, 2 * base_channels, H/16, W/16]
        up4 = self.conv4(up4)  # [B, base_channels, H/16, W/16]

        # Decoder Stage 3
        up3 = self.up3(up4)  # [B, base_channels, H/8, W/8]
        up3 = torch.cat([up3, rfb2], dim=1)  # [B, 2 * base_channels, H/8, W/8]
        up3 = self.conv3(up3)  # [B, base_channels, H/8, W/8]

        # Decoder Stage 2
        up2 = self.up2(up3)  # [B, base_channels, H/4, W/4]
        up2 = torch.cat([up2, rfb1], dim=1)  # [B, 2 * base_channels, H/4, W/4]
        up2 = self.conv2(up2)  # [B, base_channels, H/4, W/4]

        # Decoder Stage 1
        up1 = self.up1(up2)  # [B, base_channels, H/2, W/2]
        # Assuming there's a corresponding feature map from the encoder at H/2, W/2
        # If not, you can remove the skip connection or add appropriate layers
        # For simplicity, we'll skip the skip connection here
        up1 = self.conv1(torch.cat([up1, F.interpolate(rfb1, scale_factor=2, mode='bilinear', align_corners=True)],
                                   dim=1))  # [B, base_channels, H/2, W/2]

        # Final Upsampling
        out = self.final_conv(up1)  # [B, out_channels, H/2, W/2]
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)  # [B, out_channels, H, W]

        return out, agg_map_up


if __name__ == '__main__':
    net = UNet(in_channels=3, out_channels=1, base_channels=32, pretrained=True).cuda()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    output, attention = net(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Attention map shape: {attention.shape}")