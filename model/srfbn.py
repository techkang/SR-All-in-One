# Feedback Network for Image Super-Resolution
# https://arxiv.org/abs/1903.09814

import torch as t
import torch.nn as nn

from .blocks import conv_block, deconv_block


class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor, act_type, norm_type):
        super().__init__()
        arg_dict = {2: (2, 2, 6), 3: (3, 2, 7), 4: (4, 2, 8), 8: (8, 2, 12)}
        stride, padding, kernel_size = arg_dict[upscale_factor]

        self.num_groups = num_groups

        self.compress_in = conv_block(2 * num_features, num_features, kernel_size=1, act_type=act_type,
                                      norm_type=norm_type)

        self.up_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.up_tran_blocks = nn.ModuleList()
        self.down_tran_blocks = nn.ModuleList()

        for i in range(self.num_groups):
            self.up_blocks.append(deconv_block(num_features, num_features,
                                               kernel_size=kernel_size, stride=stride, padding=padding,
                                               act_type=act_type, norm_type=norm_type))
            self.down_blocks.append(conv_block(num_features, num_features,
                                               kernel_size=kernel_size, stride=stride, padding=padding,
                                               act_type=act_type, norm_type=norm_type, valid_padding=False))
            if i > 0:
                self.up_tran_blocks.append(conv_block(num_features * (i + 1), num_features,
                                                      kernel_size=1, stride=1,
                                                      act_type=act_type, norm_type=norm_type))
                self.down_tran_blocks.append(conv_block(num_features * (i + 1), num_features,
                                                        kernel_size=1, stride=1,
                                                        act_type=act_type, norm_type=norm_type))

        self.compress_out = conv_block(num_groups * num_features, num_features, kernel_size=1, act_type=act_type,
                                       norm_type=norm_type)

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = x
            self.should_reset = False

        x = t.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)

        lr_features = [x]
        hr_features = []

        for i in range(self.num_groups):
            lr_feature = t.cat(lr_features, 1)
            if i:
                lr_feature = self.up_tran_blocks[i - 1](lr_feature)
            hr_feature = self.up_blocks[i](lr_feature)

            hr_features.append(hr_feature)
            hr_feature = t.cat(hr_features, 1)
            if i:
                hr_feature = self.down_tran_blocks[i - 1](hr_feature)
            lr_feature = self.down_blocks[i](hr_feature)
            lr_features.append(lr_feature)

        output = t.cat(lr_features[1:], 1)
        output = self.compress_out(output)

        self.last_hidden = output

        return output


class SRFBN(nn.Module):
    def __init__(self, arg):
        super().__init__()
        in_channels = arg.model.in_channels
        out_channels = arg.model.out_channels
        upscale_factor = arg.upscale_factor

        num_features = arg.srfbn.num_features
        num_steps = arg.srfbn.num_steps
        num_groups = arg.srfbn.num_groups
        act_type = arg.srfbn.active
        norm_type = arg.srfbn.norm_type

        arg_dict = {2: (2, 2, 6), 3: (3, 2, 7), 4: (4, 2, 8), 8: (8, 2, 12)}
        stride, padding, kernel_size = arg_dict[arg.upscale_factor]

        self.num_steps = num_steps
        self.upscale_factor = upscale_factor

        # LR feature extraction block
        self.conv_in = conv_block(in_channels, 4 * num_features, kernel_size=3, act_type=act_type, norm_type=norm_type)
        self.feat_in = conv_block(4 * num_features, num_features, kernel_size=1, act_type=act_type, norm_type=norm_type)

        # basic block
        self.block = FeedbackBlock(num_features, num_groups, upscale_factor, act_type, norm_type)

        # reconstruction block
        self.out = deconv_block(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                                act_type='prelu', norm_type=norm_type)
        self.conv_out = conv_block(num_features, out_channels, kernel_size=3, act_type=None, norm_type=norm_type)

    def forward(self, x):
        self.block.should_reset = True

        inter_res = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)

        x = self.conv_in(x)
        x = self.feat_in(x)

        outs = []
        for _ in range(self.num_steps):
            h = self.block(x)

            h = t.add(inter_res, self.conv_out(self.out(h)))
            outs.append(h)

        return sum(outs) / len(outs)
