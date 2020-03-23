# Enhanced Deep Residual Networks for Single Image Super-Resolution
# https://arxiv.org/abs/1707.02921
import math

import torch.nn as nn
from .blocks import UpSampler

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, bias=False, bn=True, act=nn.ReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super().__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if not i:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        return self.body(x) * self.res_scale + x



class EDSR(nn.Module):
    def __init__(self, arg):
        super().__init__()
        in_channels = arg.model.in_channels
        out_channels = arg.model.out_channels
        upscale_factor = arg.upscale_factor

        self.num_features = num_features = arg.edsr.num_features
        num_blocks = arg.edsr.num_blocks
        res_scale = arg.edsr.res_scale
        conv = default_conv

        kernel_size = 3
        act = nn.ReLU()

        # define head module
        m_head = [conv(in_channels, num_features, kernel_size)]

        # define body module
        m_body = [ResBlock(conv, num_features, kernel_size, act=act, res_scale=res_scale) for _ in range(num_blocks)]
        m_body.append(conv(num_features, num_features, kernel_size))

        # define tail module
        m_tail = [
            UpSampler(conv, upscale_factor, num_features, act=False),
            conv(num_features, out_channels, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, backbone=False):
        x = self.head(x)

        res = self.body(x)
        res += x

        if backbone:
            return x

        return self.tail(x)
