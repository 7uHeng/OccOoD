import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableDeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, output_padding=0, bias=False):
        super(SeparableDeConv2d, self).__init__()

        self.depthwise_deconv = nn.ConvTranspose2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride,
            groups=in_channels,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            bias=bias
        )
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.norm_act = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.depthwise_deconv(x)
        x = self.pointwise_conv(x)
        x = self.norm_act(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.final = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]
        feat1 = self.conv1(x)
        feat6 = self.conv6(x)
        feat12 = self.conv12(x)
        feat18 = self.conv18(x)
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=True)
        out = torch.cat([feat1, feat6, feat12, feat18, global_feat], dim=1)
        return self.final(out)


class Sem_Decoder(nn.Module):
    def __init__(self, input_channel, num_classes, ratio=4):
        super().__init__()
        
        if ratio == 4:
            kernel_size = 3
            dilation = 2
            output_padding = 1
        elif ratio == 2:
            kernel_size = 3
            dilation = 1
            output_padding = 0
        else:
            raise ValueError("Only support upsample ratio of 2 or 4")

        # ASPP
        self.aspp = ASPP(input_channel, 256)


        self.upconv1 = SeparableDeConv2d(
            256, 128,
            kernel_size=kernel_size, stride=ratio // 2,
            dilation=dilation, output_padding=output_padding
        )

        self.upconv2 = SeparableDeConv2d(
            128, 64,
            kernel_size=kernel_size, stride=ratio // 2,
            dilation=1, output_padding=output_padding
        )


        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.aspp(x)     # [B, C, H, W] -> [B, 256, H, W]
        x = self.upconv1(x)  # [B, 128, 2H, 2W]
        x = self.upconv2(x)  # [B, 64, 4H, 4W]
        x = self.head(x)     # [B, num_classes, 4H, 4W]
        return x