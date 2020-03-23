# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

import numpy as np
import torch.nn as nn

import model


class MetaSR(nn.Module):
    def __init__(self, arg):
        super().__init__()

        self.backbone: model.RDN = getattr(model, arg.meta.backbone)(arg)
        self.scale = arg.upscale_factor

        num_features = self.backbone.num_features
        self.kernel_size = kernel_size = 3
        self.out_channels = arg.model.out_channels
        # self.meta_kernel = nn.Sequential(
        #     nn.Linear(1, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, arg.model.out_channels * num_features * kernel_size ** 2)
        # )

        self.conv_end = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size, 1, kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size, 1, kernel_size // 2),
        )

        self.tail = nn.Conv2d(num_features, self.out_channels, kernel_size, 1, kernel_size // 2)

    def forward(self, x, shape=None):
        if shape:
            out_h, out_w, scale = shape
        else:
            out_h, out_w, scale = x.shape[2] * self.scale, x.shape[3] * self.scale, np.array(self.scale)
        x = self.backbone(x, backbone=True)

        x = nn.functional.interpolate(x, size=(out_h, out_w))
        x = self.conv_end(x) + x
        x = self.tail(x)

        return x
