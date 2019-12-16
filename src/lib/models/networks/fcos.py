import torch
from torch import nn
from torch.nn import functional as F

from .fpn import conv_module, fill_fc_weights, dcn_module


class FCOSHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_convs, dcn=False, norm=None, activation=None, dw_conv=False,
                 scale=True, centerness=False):
        super().__init__()

        self.fpn_strides = [2, 4, 8, 16, 32]
        norm = nn.GroupNorm if norm else None

        cls_tower = []
        bbox_tower = []
        for i in range(num_convs):
            if dcn and i == num_convs - 1:
                conv_func = dcn_module
            else:
                conv_func = conv_module

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    norm=norm,
                    activation=activation,
                    dw_conv=dw_conv
                )
            )
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    norm=norm,
                    activation=activation,
                    dw_conv=dw_conv
                )
            )
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1, bias=True)
        if centerness:
            self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=True)
            fill_fc_weights(self.centerness)

        fill_fc_weights(self.cls_tower)
        fill_fc_weights(self.bbox_tower)
        fill_fc_weights(self.bbox_pred)
        fill_fc_weights(self.cls_logits)
        self.cls_logits.bias.data.fill_(-2.19)

        if scale: self.scales = nn.Parameter(torch.ones(5))

    def forward(self, inputs):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(inputs):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))

            bbox_pred = F.relu(self.bbox_pred(box_tower))
            if hasattr(self, 'scales'):
                bbox_pred *= self.scales[l]
            bbox_reg.append(bbox_pred * self.fpn_strides[l])
            if hasattr(self, 'centerness'):
                centerness.append(self.centerness(box_tower))
        return logits, bbox_reg, centerness
