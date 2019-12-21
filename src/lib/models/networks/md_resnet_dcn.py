# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .fpn import *
from .fcos import FCOSHead
from .ttfnet import TTFHead

BN = nn.BatchNorm2d
# BN = nn.SyncBatchNorm
# BN = nn.GroupNorm

logger = logging.getLogger(__name__)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BN(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BN(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv, down_ratio=4, width_factor=1., deconv_k=4, dcn=False,
                 dw_conv=False, bottom_up=0, skip=False, mode=1):
        assert down_ratio in [2, 4, 8, 16]
        self.heads = heads
        self.deconv_with_bias = False
        if isinstance(width_factor, float):
            width_factor = [width_factor] * 5
        assert len(width_factor) == 5
        self.inplanes = int(64 * width_factor[0])

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BN(self.inplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], factor=width_factor[1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, factor=width_factor[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, factor=width_factor[3])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, factor=width_factor[4])
        up_num = int(math.log2(32 / down_ratio))
        filters = [int(f * block.expansion * width_factor[3 - i]) for i, f in enumerate([64, 64, 128, 256, 512])]

        stack = 1
        dims = filters[4-up_num] if isinstance(heads, int) else 0
        # self.fpn = FPN(filters, 0, 4 - up_num, 5, deconv_k, BN, nn.ReLU, dcn, dw_conv, mode)
        self.fpn = StackBiFPN(filters, dims, 4 - up_num, 5, stack, BN, nn.ReLU, dw_conv, deconv_k, dcn, skip, bottom_up,
                              mode)
        # self.fpn = MdFPN(filters, 0, 4 - up_num, 5, dcn, bottom_up, deconv_k, BN, nn.ReLU, dw_conv, skip, mode)

        if isinstance(heads, int):
            self.fcos = FCOSHead(dims, heads, head_conv, dcn, False, nn.ReLU, dw_conv, True, True)
        elif 'ttf' in heads:
            self.ttf = TTFHead(filters[4 - up_num], heads['ttf'], head_conv)
        else:
            head_conv = int(head_conv * width_factor[-1])
            for head in self.heads:
                classes = self.heads[head]
                if head_conv > 0:
                    fc = nn.Sequential(
                        nn.Conv2d(filters[4 - up_num], head_conv,
                                  kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, classes,
                                  kernel_size=1, stride=1,
                                  padding=0, bias=True))
                    if 'hm' in head or 'obj' in head:
                        fc[-1].bias.data.fill_(-2.19)
                    else:
                        fill_fc_weights(fc)
                else:
                    fc = nn.Conv2d(filters[4 - up_num], classes,
                                   kernel_size=1, stride=1,
                                   padding=0, bias=True)
                    if 'hm' in head or 'obj' in head:
                        fc.bias.data.fill_(-2.19)
                    else:
                        fill_fc_weights(fc)
                self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1, factor=1.):
        downsample = None
        planes = int(planes * factor)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BN(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        stage = [x]
        x = self.maxpool(x)

        x = self.layer1(x)
        stage.append(x)
        x = self.layer2(x)
        stage.append(x)
        x = self.layer3(x)
        stage.append(x)
        x = self.layer4(x)
        stage.append(x)

        x = self.fpn(stage)

        if hasattr(self, 'fcos'):
            ret = self.fcos(x)
            return [ret]

        x = x[0]
        if hasattr(self, 'ttf'):
            ret = self.ttf(x)
            return [ret]
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            # nn.init.normal_(self.conv1.weight, std=0.001)
            torch.nn.init.kaiming_normal_(self.conv1.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(self.conv1.weight.data)
            for i in range(1, 5): fill_fc_weights(getattr(self, 'layer%d' % i))
            for m in self.modules():
                if isinstance(m, BN):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_conv=64, down_ratio=4, width_factor=1., **kwargs):
    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, heads, head_conv=head_conv, down_ratio=down_ratio,
                       width_factor=width_factor, **kwargs)
    model.init_weights(num_layers, np.all(np.array(width_factor) == 1))
    # model.init_weights(num_layers, False)
    return model
