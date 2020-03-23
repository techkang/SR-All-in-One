# Learning Enriched Features for Real Image Restoration and Enhancement
# https://arxiv.org/abs/2003.06792

import torch as t
import torch.nn as nn

from .rcan import CALayer


class SKFF(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.down_sample = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.PReLU(),
        )

        self.up_samples = nn.ModuleList(
            [nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True) for _ in range(3)])
        self.softmax = nn.Softmax()

    def forward(self, f_1, f_2, f_3):
        x = f_1 + f_2 + f_3
        x = self.GAP(x)
        x = self.down_sample(x)
        feature = t.stack([module(x) for module in self.up_samples], dim=1).softmax(2)
        s_1, s_2, s_3 = feature.transpose(0, 1).contiguous()
        return f_1 * s_1 + f_2 * s_2 + f_3 * s_3


class SALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.CA = CALayer(channel=channel, reduction=8)
        self.sa_conv = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        sa_feature = t.cat([x.max(1, keepdim=True)[0], x.mean(1, keepdim=True)], 1)
        sa_feature = self.sa_conv(sa_feature)
        return x * sa_feature


class DAU(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        )
        self.SA = SALayer(channel)
        self.CA = CALayer(channel, reduction=8)
        self.tail = nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1)

    def forward(self, x):
        feature = self.head(x)
        feature = t.cat([self.SA(feature), self.CA(feature)], 1)
        feature = self.tail(feature)
        return x + feature


class Sample(nn.Module):
    def __init__(self, channel, scale=1 / 2):
        super().__init__()
        if scale == 1 / 2:
            self.sample = nn.Upsample(scale_factor=1 / 2, mode='bicubic', align_corners=False)
        else:
            self.sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.PReLU(),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.PReLU(),
            self.sample,
            nn.Conv2d(channel, int(channel / scale), 1)
        )
        self.down_conv = nn.Sequential(
            self.sample,
            nn.Conv2d(channel, int(channel / scale), 1)
        )

    def forward(self, x):
        return self.up_conv(x) + self.down_conv(x)


class MRB(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.down_2x = nn.ModuleList([Sample(channel * i, 1 / 2) for i in [1, 1, 2]])
        self.down_4x = nn.ModuleList(
            [nn.Sequential(Sample(channel, 1 / 2), Sample(channel * 2, 1 / 2)) for _ in range(2)])
        self.up_2x = nn.ModuleList([Sample(channel * i, 2) for i in [2, 4, 2]])
        self.up_4x = nn.ModuleList([nn.Sequential(Sample(channel * 4, 2), Sample(channel * 2, 2)) for _ in range(2)])
        self.DAUs_1 = nn.ModuleList([DAU(channel * i) for i in [1, 2, 4]])
        self.DAUs_2 = nn.ModuleList([DAU(channel * i) for i in [1, 2, 4]])
        self.SKFF = nn.ModuleList(SKFF(i) for i in [channel, channel * 2, channel * 4, channel])
        self.conv = nn.Conv2d(channel, channel, 3, padding=1)

    def forward(self, x):
        top = self.DAUs_1[0](x)
        middle = self.DAUs_1[1](self.down_2x[0](x))
        down = self.DAUs_1[2](self.down_4x[0](x))
        top_2x_down = self.down_2x[1](top)
        top_4x_down = self.down_4x[1](top)
        middle_2x_up = self.up_2x[0](middle)
        middle_2x_down = self.down_2x[2](middle)
        down_2x_up = self.up_2x[1](down)
        down_4x_up = self.up_4x[0](down)
        top = self.DAUs_2[0](self.SKFF[0](top, middle_2x_up, down_4x_up))
        middle = self.DAUs_2[1](self.SKFF[1](top_2x_down, middle, down_2x_up))
        down = self.DAUs_2[2](self.SKFF[2](top_4x_down, middle_2x_down, down))
        middle = self.up_2x[2](middle)
        down = self.up_4x[1](down)
        feature = self.SKFF[3](top, middle, down)
        x = self.conv(feature) + x
        return x


class RRG(nn.Module):
    def __init__(self, channel, num_blocks):
        super().__init__()
        self.head = nn.Conv2d(channel, channel, 3, padding=1)
        self.tail = nn.Conv2d(channel, channel, 3, padding=1)
        self.MRBs = nn.Sequential(*[MRB(channel) for _ in range(num_blocks)])

    def forward(self, x):
        feature = self.tail(self.MRBs(self.head(x)))
        return x + feature


class MIRNet(nn.Module):
    def __init__(self, arg):
        super().__init__()
        channel = arg.mirnet.num_features
        num_blocks = arg.mirnet.num_blocks
        num_groups = arg.mirnet.num_groups
        self.scale = arg.upscale_factor

        self.head = nn.Conv2d(3, channel, 3, padding=1)
        self.tail = nn.Conv2d(channel, 3, 3, padding=1)
        self.RRGs = nn.Sequential(*[RRG(channel, num_blocks) for _ in range(num_groups)])

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
        feature = self.tail(self.RRGs(self.head(x)))
        return x + feature
