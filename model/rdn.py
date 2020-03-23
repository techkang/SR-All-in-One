# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

import torch as t
import torch.nn as nn


class RDBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class RDB(nn.Module):
    def __init__(self, num_features, num_layers):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(RDBConv((i + 1) * num_features, num_features))
        self.layers = nn.ModuleList(layers)

        # Local Feature Fusion
        self.LFF = nn.Conv2d((num_layers + 1) * num_features, num_features, 1, padding=0, stride=1)

    def forward(self, x):
        out = [x]
        for layer in self.layers:
            x = layer(t.cat(out, 1))
            out.append(x)

        return self.LFF(t.cat(out, 1)) + x


class RDN(nn.Module):
    def __init__(self, arg):
        super().__init__()
        in_channels = arg.model.in_channels
        out_channels = arg.model.out_channels
        upscale_factor = arg.upscale_factor

        self.num_features = num_features = arg.rdn.num_features
        num_blocks = arg.rdn.num_blocks
        self.num_blocks = num_blocks
        num_layers = arg.rdn.num_layers

        kernel_size = 3

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(in_channels, num_features, kernel_size, padding=kernel_size // 2, stride=1)
        self.SFENet2 = nn.Conv2d(num_features, num_features, kernel_size, padding=kernel_size // 2, stride=1)

        # Residual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList([RDB(num_features, num_layers) for _ in range(num_blocks)])

        # Global Feature Fusion
        self.GFF = nn.Sequential(
            nn.Conv2d(num_blocks * num_features, num_features, 1, padding=0, stride=1),
            nn.Conv2d(num_features, num_features, kernel_size, padding=kernel_size // 2, stride=1)
        )

        # Up-sampling net
        if upscale_factor == 2 or upscale_factor == 3:
            self.UPNet = nn.Sequential(
                nn.Conv2d(num_features, num_features * upscale_factor ** 2, kernel_size, padding=kernel_size // 2,
                          stride=1),
                nn.PixelShuffle(upscale_factor),
                nn.Conv2d(num_features, out_channels, kernel_size, padding=kernel_size // 2, stride=1)
            )
        elif upscale_factor == 4:
            self.UPNet = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, kernel_size, padding=kernel_size // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, num_features * 4, kernel_size, padding=kernel_size // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, out_channels, kernel_size, padding=kernel_size // 2, stride=1)
            )
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x, backbone=False):

        f_1 = self.SFENet1(x)
        x = self.SFENet2(f_1)

        out = []
        for i in range(self.num_blocks):
            x = self.RDBs[i](x)
            out.append(x)

        x = self.GFF(t.cat(out, 1))
        x += f_1
        if backbone:
            return x

        return self.UPNet(x)
