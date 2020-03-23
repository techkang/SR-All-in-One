# Denoising Prior Driven Deep Neural Network for Image Restoration
# https://arxiv.org/abs/1801.06756

import torch as t
import torch.nn as nn
import torch.nn.functional as F


class DCNNDenoiser(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder_1 = self.make_blocks(in_channels)
        self.encoder_2 = self.make_blocks()
        self.encoder_3 = self.make_blocks()
        self.encoder_4 = self.make_blocks()
        self.bridge = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.decoder_4 = self.make_blocks(128, up_sample=True)
        self.decoder_3 = self.make_blocks(128, up_sample=True)
        self.decoder_2 = self.make_blocks(128, up_sample=True)
        self.decoder_1 = self.make_blocks(128)
        self.conv_end = nn.Conv2d(64, 1, 3, 1, 1, bias=False)

    def make_blocks(self, in_channels=64, up_sample=False):
        same_module = 4
        if in_channels == 1 or in_channels == 3:
            seq = [nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False), nn.ReLU()]
            same_module = 3
        elif in_channels == 64:
            seq = [nn.Conv2d(64, 64, 3, 2, 1, bias=False), nn.ReLU()]
            same_module = 3
        elif in_channels == 128:
            seq = [nn.Conv2d(128, 64, 1, bias=False), nn.ReLU()]
        else:
            raise ValueError(f'in channels is {in_channels}, but only support'
                             f'[1, 3, 64, 128]')
        for _ in range(same_module):
            seq += [nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.ReLU()]
        if up_sample:
            seq += [nn.ConvTranspose2d(64, 64, 2, 2, bias=False)]
        return nn.Sequential(*seq)

    def forward(self, noise_image):
        f_1 = self.encoder_1(noise_image)
        f_2 = self.encoder_2(f_1)
        f_3 = self.encoder_2(f_2)
        f_4 = self.encoder_2(f_3)
        u_f_4 = t.relu(self.bridge(f_4))
        u_f_3 = self.decoder_4(t.cat((u_f_4, f_4), 1))
        u_f_2 = self.decoder_3(t.cat((u_f_3, f_3), 1))
        u_f_1 = self.decoder_2(t.cat((u_f_2, f_2), 1))
        u_f_0 = self.decoder_1(t.cat((u_f_1, f_1), 1))
        u_f = self.conv_end(u_f_0)
        return u_f + noise_image


class DPDNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_channels = cfg.model.in_channels
        self.denoiser = DCNNDenoiser(self.in_channels)
        self.scale = cfg.upscale_factor
        self.up_sample_scale = cfg.upscale_factor
        self.multi_scale = bool(cfg.solver.multi_scale)
        # self.interpolation = self.get_interpolation(cfg.datasets.interpolation)
        self.max_iter = cfg.dpdnn.iteration
        self.a_bar = self.a_bar_net()
        self.delta = nn.Parameter(t.ones(self.max_iter) * 0.1, True)
        self.eta = nn.Parameter(t.ones(self.max_iter) * 0.9, True)
        self.up_sample = nn.ConvTranspose2d(self.in_channels, self.in_channels, 2 * self.scale, self.scale,
                                            self.scale // 2, bias=False)
        self.up_sample_2 = nn.ConvTranspose2d(self.in_channels, self.in_channels, 2 * self.scale,
                                              self.scale, self.scale // 2, bias=False)
        self.down_sample = nn.Conv2d(self.in_channels, self.in_channels, 2 * self.scale, self.scale,
                                     self.scale // 2, bias=False)

    def a_bar_net(self):
        return nn.Conv2d(self.in_channels, self.in_channels, 5, 1, 2, bias=False)

    def iterative(self, noise_image):
        x = self.interpolation(noise_image)
        top = x
        for i in range(self.max_iter):
            top = top * self.delta[i] * self.eta[i]
            down = self.a_bar(x)
            middle = self.denoiser(x) * self.delta[i]
            x = middle + top + down
        return x

    def tf_version(self, noise_image):
        x = self.up_sample(noise_image)
        # x = F.interpolate(noise_image, scale_factor=self.up_sample_scale, mode='bicubic', align_corners=False)
        for i in range(self.max_iter):
            conv_out = self.denoiser(x)
            down = self.down_sample(x)
            err1 = self.up_sample_2(down - noise_image)
            # down = F.interpolate(x, scale_factor=1 / self.up_sample_scale, mode='bicubic', align_corners=False)
            # err1 = F.interpolate(down - noise_image, scale_factor=self.up_sample_scale, mode='bicubic',
            #                      align_corners=False)
            err2 = x - conv_out
            x = x - self.delta[i] * (err1 + self.eta[i] * err2)

        return x

    def forward(self, x):
        pred = self.tf_version(x)
        return pred
