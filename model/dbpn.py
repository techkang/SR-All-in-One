# Deep Back-Projection Networks for Super-Resolution
# https://arxiv.org/abs/1803.02735

import torch as t
import torch.nn as nn

from .blocks import conv_block, deconv_block


class UpProjBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, valid_padding=False, padding=0, norm_type=None,
                 act_type='prelu'):
        super().__init__()

        self.deconv_1 = deconv_block(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                     norm_type=norm_type, act_type=act_type)

        self.conv_1 = conv_block(out_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 valid_padding=valid_padding, norm_type=norm_type, act_type=act_type)

        self.deconv_2 = deconv_block(out_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                     norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        h_0_t = self.deconv_1(x)
        l_0_t = self.conv_1(h_0_t)
        h_1_t = self.deconv_2(l_0_t - x)

        return h_0_t + h_1_t


class DownProjBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, valid_padding=False, padding=0, norm_type=None,
                 act_type='prelu'):
        super().__init__()
        self.conv_1 = conv_block(in_channel, out_channel, kernel_size, stride=stride, valid_padding=valid_padding,
                                 padding=padding, norm_type=norm_type, act_type=act_type)
        self.deconv_1 = deconv_block(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                     norm_type=norm_type, act_type=act_type)
        self.conv_2 = conv_block(in_channel, out_channel, kernel_size, stride=stride, valid_padding=valid_padding,
                                 padding=padding, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        l_0_t = self.conv_1(x)
        h_0_t = self.deconv_1(l_0_t)
        l_1_t = self.conv_2(h_0_t - x)

        return l_0_t + l_1_t


class DenseBackProjBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, bp_stages, stride=1, padding=0, norm_type=None,
                 act_type='prelu'):
        super().__init__()

        self.up_proj = nn.ModuleList()
        self.down_proj = nn.ModuleList()
        self.bp_stages = bp_stages
        self.up_proj.append(UpProjBlock(in_channel, out_channel, kernel_size, stride=stride, valid_padding=False,
                                        padding=padding, norm_type=norm_type, act_type=act_type))

        for index in range(self.bp_stages - 1):
            if index < 1:
                self.up_proj.append(
                    UpProjBlock(out_channel, out_channel, kernel_size, stride=stride, valid_padding=False,
                                padding=padding, norm_type=norm_type, act_type=act_type))
            else:
                uc = conv_block(out_channel * (index + 1), out_channel, kernel_size=1, norm_type=norm_type,
                                act_type=act_type)
                u = UpProjBlock(out_channel, out_channel, kernel_size, stride=stride, valid_padding=False,
                                padding=padding, norm_type=norm_type, act_type=act_type)
                self.up_proj.append(nn.Sequential(uc, u))

            if index < 1:
                self.down_proj.append(
                    DownProjBlock(out_channel, out_channel, kernel_size, stride=stride, valid_padding=False,
                                  padding=padding, norm_type=norm_type, act_type=act_type))
            else:
                dc = conv_block(out_channel * (index + 1), out_channel, kernel_size=1, norm_type=norm_type,
                                act_type=act_type)
                d = DownProjBlock(out_channel, out_channel, kernel_size, stride=stride, valid_padding=False,
                                  padding=padding, norm_type=norm_type, act_type=act_type)
                self.down_proj.append(nn.Sequential(dc, d))

    def forward(self, x):
        low_features = []
        high_features = []

        high = self.up_proj[0](x)
        high_features.append(high)

        for index in range(self.bp_stages - 1):
            high_concat = t.cat(high_features, 1)
            low = self.down_proj[index](high_concat)
            low_features.append(low)
            low_concat = t.cat(low_features, 1)
            high = self.up_proj[index + 1](low_concat)
            high_features.append(high)

        output = t.cat(high_features, 1)
        return output


class DBPN(nn.Module):
    def __init__(self, arg):
        super().__init__()
        in_channels = arg.model.in_channels
        out_channels = arg.model.out_channels

        num_features = arg.dbpn.num_features
        bp_stages = arg.dbpn.num_blocks
        norm_type = arg.dbpn.norm_type
        act_type = arg.dbpn.active

        arg_dict = {2: (2, 2, 6), 4: (4, 2, 8), 8: (8, 2, 12)}
        stride, padding, projection_filter = arg_dict[arg.upscale_factor]

        feature_extract_1 = conv_block(in_channels, 128, kernel_size=3, norm_type=norm_type, act_type=act_type)
        feature_extract_2 = conv_block(128, num_features, kernel_size=1, norm_type=norm_type, act_type=act_type)

        bp_units = []
        for _ in range(bp_stages - 1):
            bp_units.extend(
                [UpProjBlock(num_features, num_features, projection_filter, stride=stride, valid_padding=False,
                             padding=padding, norm_type=norm_type, act_type=act_type),
                 DownProjBlock(num_features, num_features, projection_filter, stride=stride, valid_padding=False,
                               padding=padding, norm_type=norm_type, act_type=act_type)])

        last_bp_unit = UpProjBlock(num_features, num_features, projection_filter, stride=stride, valid_padding=False,
                                   padding=padding, norm_type=norm_type, act_type=act_type)
        conv_hr = conv_block(num_features, out_channels, kernel_size=1, norm_type=None, act_type=None)

        self.network = nn.Sequential(feature_extract_1, feature_extract_2, *bp_units, last_bp_unit, conv_hr)

    def forward(self, x):
        return self.network(x)


class DDBPN(nn.Module):
    def __init__(self, arg):
        super().__init__()
        in_channels = arg.model.in_channels
        out_channels = arg.model.out_channels
        num_features = arg.dbpn.num_features
        bp_stages = arg.dbpn.num_blocks
        norm_type = arg.dbpn.norm_type
        act_type = arg.dbpn.active

        arg_dict = {2: (2, 2, 6), 4: (4, 2, 8), 8: (8, 2, 12)}
        stride, padding, projection_filter = arg_dict[arg.upscale_factor]

        feature_extract_1 = conv_block(in_channels, 256, kernel_size=3, norm_type=norm_type, act_type=act_type)
        feature_extract_2 = conv_block(256, num_features, kernel_size=1, norm_type=norm_type, act_type=act_type)

        bp_units = DenseBackProjBlock(num_features, num_features, projection_filter, bp_stages, stride=stride,
                                      padding=padding, norm_type=norm_type, act_type=act_type)

        conv_hr = conv_block(num_features * bp_stages, out_channels, kernel_size=3, norm_type=None, act_type=None)

        self.network = nn.Sequential(feature_extract_1, feature_extract_2, bp_units, conv_hr)

    def forward(self, x):
        return self.network(x)
