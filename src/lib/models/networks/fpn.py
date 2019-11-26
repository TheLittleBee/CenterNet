import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

from .DCNv2.dcn_v2 import DCN

BN_MOMENTUM = 0.1


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(fc):
    for m in fc.modules():
        if isinstance(m, nn.Conv2d):
            # nn.init.normal_(m.weight, std=0.001)
            torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def norm_module(norm, channels):
    if norm == nn.GroupNorm:
        return nn.GroupNorm(16, channels)
    return norm(channels, momentum=BN_MOMENTUM)


def act_module(activation):
    if activation == nn.ReLU:
        return nn.ReLU(inplace=True)
    return activation()


def conv_module(in_channels, out_channels, kernel_size, stride=1, norm=None, activation=None, dw_conv=False):
    padding = (kernel_size - 1) // 2
    layers = []
    if dw_conv:
        layers.append(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=norm is None))
        if norm: layers.append(norm_module(norm, in_channels))
        if activation: layers.append(act_module(activation))
        layers.append(nn.Conv2d(in_channels, out_channels, 1, bias=norm is None))
        if norm: layers.append(norm_module(norm, out_channels))
    else:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 1, 1, bias=norm is None))
        if norm: layers.append(norm_module(norm, out_channels))
        if activation: layers.append(act_module(activation))
    return nn.Sequential(*layers)


def deconv_module(in_channels, out_channels, kernel_size, stride=2, norm=None, activation=None, dw_conv=False):
    assert kernel_size in [2, 3, 4]
    n = int(math.log2(stride))
    if kernel_size == 2:
        padding, output_padding = 0, 0
    elif kernel_size == 3:
        padding, output_padding = 1, 1
    else:
        padding, output_padding = 1, 0
    layers = []
    if dw_conv:
        for _ in range(n):
            layers.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size, 2, padding, output_padding,
                                             groups=in_channels, bias=norm is None))
            if norm: layers.append(norm_module(norm, in_channels))
            if activation: layers.append(act_module(activation))
        layers.append(conv_module(in_channels, out_channels, 1, 1, norm))
    else:
        for _ in range(n):
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 2, padding, output_padding, 1,
                                             bias=norm is None))
            if norm: layers.append(norm_module(norm, out_channels))
            if activation: layers.append(act_module(activation))
            in_channels = out_channels
    return nn.Sequential(*layers)


class FPN(nn.Module):
    def __init__(self, num_filters, dims=256, min_level=2, max_level=5, P2only=False, bottom_up=False,
                 deconv_k=0, norm=None, activation=None, dw_conv=False):
        super().__init__()
        self.P2only = P2only
        self.bottom_up = bottom_up
        self.norm = norm

        self.dim_out = fpn_dim = dims
        self.num_stages = 5 - min_level

        self.top_down_ups = nn.ModuleList() if deconv_k else None
        self.top_down_nodes = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        self.lateral_top = conv_module(num_filters[-1], fpn_dim, 1, 1, norm, activation)

        # build up, lateral conv, fusion node in top down way
        for i in range(4, min_level - 1, -1):
            if deconv_k:
                self.top_down_ups.append(deconv_module(fpn_dim, fpn_dim, deconv_k, 2, norm, activation, dw_conv))
            self.lateral_convs.append(conv_module(num_filters[i - 1], fpn_dim, 1, 1, norm, activation))
            self.top_down_nodes.append(conv_module(fpn_dim, fpn_dim, 3, 1, norm, activation, dw_conv))

        # build stride conv, fusion node in bottom up way, for PANet
        if self.bottom_up:
            self.bottom_up_convs = nn.ModuleList()
            self.bottom_up_nodes = nn.ModuleList()
            for i in range(self.num_stages - 1):
                self.bottom_up_convs.append(conv_module(fpn_dim, fpn_dim, 3, 2, norm, activation, dw_conv))
                self.bottom_up_nodes.append(conv_module(fpn_dim, self.dim_out, 3, 1, norm, activation, dw_conv))

        # build extra conv append to the output, which way in RetinaNet
        self.extra_convs = None
        if max_level > 5:
            self.extra_convs = nn.ModuleList()
            for i in range(5, max_level):
                self.extra_convs.append(conv_module(fpn_dim, fpn_dim, 3, 2, norm, activation, dw_conv))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                fill_up_weights(m)
            elif isinstance(m, nn.Conv2d):
                fill_fc_weights(m)
            elif self.norm is not None and isinstance(m, self.norm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """

        Return: top down order, from low to high resolution
        """
        fpn_inner_blobs = [self.lateral_top(inputs[-1])]
        for i in range(self.num_stages - 1):
            if self.top_down_ups:
                up = self.top_down_ups[i](fpn_inner_blobs[-1])
            else:
                up = F.interpolate(fpn_inner_blobs[-1], scale_factor=2, mode='nearest')
            fpn_inner_blobs.append(self.top_down_nodes[i](self.lateral_convs[i](inputs[-2 - i]) + up))
        fpn_output_blobs = []
        if self.bottom_up:
            fpn_output_blobs.append(fpn_inner_blobs[-1])
            for i in range(2, self.num_stages + 1):
                fpn_tmp = self.bottom_up_convs[i - 2](fpn_output_blobs[0])
                fpn_tmp = fpn_tmp + fpn_inner_blobs[self.num_stages - i]
                fpn_tmp = self.bottom_up_nodes[i - 2](fpn_tmp)
                fpn_output_blobs.insert(0, fpn_tmp)
        else:
            fpn_output_blobs = fpn_inner_blobs
        if self.extra_convs is not None:
            for l in self.extra_convs:
                fpn_output_blobs.insert(0, l(fpn_output_blobs[0]))

        if self.P2only:
            return fpn_output_blobs[0]
        else:
            return tuple(fpn_output_blobs)


class StackBiFPN(nn.Module):
    def __init__(self, num_filters, dims, min_level=2, max_level=5, stack=1, norm=None, activation=None, dw_conv=False,
                 deconv_k=0, weighted=True, skip=True, bottom_up=True, se=False):
        super().__init__()
        if not weighted and not bottom_up:
            print('Stack original FPN for %d times.' % stack)
        if not weighted and not skip and bottom_up:
            print('Stack PANet for %d times.' % stack)
        if weighted and skip and bottom_up and not se:
            print('Stack BiFPN for %d times' % stack)
        self.norm = norm
        self.dim_out = fpn_dim = dims
        self.levels = 5 - min_level
        self.num_out = max_level - min_level
        self.stack = stack

        self.lateral_convs = nn.ModuleList()
        self.stack_bifpn = nn.ModuleList()

        # build lateral conv to make all inputs have same channels
        for i in range(min_level, 5):
            self.lateral_convs.append(conv_module(num_filters[i], fpn_dim, 1, 1, norm, activation))
        for i in range(stack):
            self.stack_bifpn.append(
                BiFPN(fpn_dim, self.levels, norm, activation, dw_conv, deconv_k, weighted, skip, bottom_up, se))

        # build extra convs append on output, which way in RetinaNet
        self.extra_convs = None
        if max_level > 5:
            self.extra_convs = nn.ModuleList()
            for i in range(5, max_level):
                self.extra_convs.append(conv_module(fpn_dim, fpn_dim, 3, 2, norm, activation, dw_conv))

    def _init_weights(self):
        for l in [self.lateral_convs, self.extra_convs]:
            if l is not None:
                for m in l.modules():
                    if isinstance(m, nn.Conv2d):
                        fill_fc_weights(m)
        for m in self.modules():
            if self.norm is not None and isinstance(m, self.norm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        laterals = []
        for i in range(self.levels, 0, -1):
            laterals.append(self.lateral_convs[self.levels - i](inputs[-i]))
        outs = self.stack_bifpn[0](laterals)
        for i in range(1, self.stack):
            outs = self.stack_bifpn[i](outs)
        # get output from low to high resolution
        outs = outs[::-1]
        if self.num_out > self.levels:
            for i in range(self.levels, self.num_out):
                outs.insert(0, self.extra_convs[i - self.levels](outs[0]))
        return tuple(outs)


class BiFPN(nn.Module):
    def __init__(self, fpn_dim, levels, norm=None, activation=None, dw_conv=False, deconv_k=0, weighted=True, skip=True,
                 bottom_up=True, se=False):
        if weighted and not se and not skip:
            raise Exception('not suppot')
        super(BiFPN, self).__init__()
        self.levels = levels
        self.bottom_up = bottom_up
        self.skip = skip
        self.w1 = nn.Parameter(torch.rand(2, levels)) if weighted and not se else None
        self.w2 = nn.Parameter(torch.rand(3, levels - 2)) if weighted and not se else None

        self.top_down_ups = nn.ModuleList() if deconv_k else None
        self.top_down_nodes = nn.ModuleList()

        for i in range(levels - 1):
            if deconv_k:
                self.top_down_ups.append(deconv_module(fpn_dim, fpn_dim, deconv_k, 2, norm, activation, dw_conv))
            self.top_down_nodes.append(conv_module(fpn_dim, fpn_dim, 3, 1, norm, activation, dw_conv))

        if self.bottom_up:
            self.bottom_up_convs = nn.ModuleList() if deconv_k else None
            self.bottom_up_nodes = nn.ModuleList()
            for i in range(self.levels - 1):
                if deconv_k: self.bottom_up_convs.append(conv_module(fpn_dim, fpn_dim, 3, 2, norm, activation, dw_conv))
                self.bottom_up_nodes.append(conv_module(fpn_dim, fpn_dim, 3, 1, norm, activation, dw_conv))

        # build attention conv
        if weighted and se:
            self.se1 = nn.ModuleList()
            self.se2 = nn.ModuleList() if bottom_up else None
            for i in range(levels - 1):
                self.se1.append(conv_module(2 * fpn_dim, 2, 1, 1, norm, nn.Sigmoid))
            if bottom_up:
                for i in range(levels - 2):
                    if skip:
                        self.se2.append(conv_module(3 * fpn_dim, 3, 1, 1, norm, nn.Sigmoid))
                    else:
                        self.se2.append(conv_module(2 * fpn_dim, 2, 1, 1, norm, nn.Sigmoid))
                self.se2.append(conv_module(2 * fpn_dim, 2, 1, 1, norm, nn.Sigmoid))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                fill_up_weights(m)
            elif isinstance(m, nn.Conv2d):
                fill_fc_weights(m)

    def forward(self, inputs):
        """

        Note: inputs and return should have same order, like from high to low resolution
        """
        assert len(inputs) == self.levels
        fpn_inner_blobs = [inputs[-1]]
        for i in range(self.levels - 1):
            if self.top_down_ups:
                up = self.top_down_ups[i](fpn_inner_blobs[-1])
            else:
                up = F.interpolate(fpn_inner_blobs[-1], scale_factor=2, mode='nearest')
            if self.w1 is not None:
                fusion = (self.w1[0, i] * inputs[-2 - i] + self.w1[1, i] * up) / (self.w1[0, i] + self.w1[1, i] + 1e-5)
            elif hasattr(self, 'se1'):
                fusion_avg = torch.cat((F.adaptive_avg_pool2d(inputs[-2 - i], 1), F.adaptive_avg_pool2d(up, 1)), 1)
                fusion_w = self.se1[i](fusion_avg)
                fusion = (fusion_w[:, 0:1] * inputs[-2 - i] + fusion_w[:, 1:2] * up) / (
                        torch.sum(fusion_w, 1, keepdim=True) + 1e-5)
            else:
                fusion = inputs[-2 - i] + up
            fpn_inner_blobs.append(self.top_down_nodes[i](fusion))

        fpn_output_blobs = []
        # bottom up way has two cases: layer with three nodes; layer with two nodes (the top layer)
        if self.bottom_up:
            fpn_output_blobs.append(fpn_inner_blobs[-1])
            for i in range(self.levels - 2):
                if self.bottom_up_convs:
                    fpn_tmp = self.bottom_up_convs[i](fpn_output_blobs[-1])
                else:
                    fpn_tmp = F.avg_pool2d(fpn_output_blobs[-1], 3, 2, 1)
                if self.w2 is not None:
                    fusion = (self.w2[0, i] * inputs[i + 1] + self.w2[1, i] * fpn_inner_blobs[-2 - i] +
                              self.w2[2, i] * fpn_tmp) / (self.w2[0, i] + self.w2[1, i] + self.w2[2, i] + 1e-5)
                elif hasattr(self, 'se2'):
                    if self.skip:
                        fusion_avg = torch.cat((F.adaptive_avg_pool2d(inputs[i + 1], 1),
                                                F.adaptive_avg_pool2d(fpn_inner_blobs[-2 - i], 1),
                                                F.adaptive_avg_pool2d(fpn_tmp, 1)), 1)
                        fusion_w = self.se2[i](fusion_avg)
                        fusion = (fusion_w[:, 0:1] * inputs[i + 1] + fusion_w[:, 1:2] * fpn_inner_blobs[-2 - i] +
                                  fusion_w[:, 2:3] * fpn_tmp) / (torch.sum(fusion_w, 1, keepdim=True) + 1e-5)
                    else:
                        fusion_avg = torch.cat((F.adaptive_avg_pool2d(fpn_inner_blobs[-2 - i], 1),
                                                F.adaptive_avg_pool2d(fpn_tmp, 1)), 1)
                        fusion_w = self.se2[-1](fusion_avg)
                        fusion = (fusion_w[:, 0:1] * fpn_inner_blobs[-2 - i] + fusion_w[:, 1:2] * fpn_tmp) / (
                                torch.sum(fusion_w, 1, keepdim=True) + 1e-5)
                else:
                    fusion = fpn_inner_blobs[-2 - i] + fpn_tmp
                fpn_tmp = self.bottom_up_nodes[i](fusion)
                fpn_output_blobs.append(fpn_tmp)
            if self.bottom_up_convs:
                fpn_tmp = self.bottom_up_convs[-1](fpn_output_blobs[-1])
            else:
                fpn_tmp = F.avg_pool2d(fpn_output_blobs[-1], 3, 2, 1)
            if self.w1 is not None:
                fusion = (self.w1[0, -1] * fpn_inner_blobs[0] + self.w1[1, -1] * fpn_tmp) / (
                        self.w1[0, -1] + self.w1[1, -1] + 1e-5)
            elif hasattr(self, 'se2'):
                fusion_avg = torch.cat(
                    (F.adaptive_avg_pool2d(fpn_inner_blobs[0], 1), F.adaptive_avg_pool2d(fpn_tmp, 1)), 1)
                fusion_w = self.se2[-1](fusion_avg)
                fusion = (fusion_w[:, 0:1] * fpn_inner_blobs[0] + fusion_w[:, 1:2] * fpn_tmp) / (
                        torch.sum(fusion_w, 1, keepdim=True) + 1e-5)
            else:
                fusion = fpn_inner_blobs[0] + fpn_tmp
            fpn_output_blobs.append(self.bottom_up_nodes[-1](fusion))
        else:
            fpn_output_blobs = fpn_inner_blobs[::-1]
        return fpn_output_blobs


def dcn_module(in_channels, out_channels, kernel_size, stride=1, norm=None, activation=None, dw_conv=False):
    layers = []
    if dw_conv:
        layers.append(DCN(in_channels, in_channels, 3, 1, 1, 1, in_channels))
        if norm: layers.append(norm_module(norm, in_channels))
        if activation: layers.append(act_module(activation))
        layers.append(conv_module(in_channels, out_channels, 1, 1, norm))
    else:
        layers.append(DCN(in_channels, out_channels, 3, 1, 1, 1, 1))
        if norm: layers.append(norm_module(norm, out_channels))
        if activation: layers.append(act_module(activation))
    return nn.Sequential(*layers)


class MdFPN(nn.Module):
    def __init__(self, num_filters, dims=64, min_level=0, max_level=5, dcn=True, bottom_up=2, deconv_k=0,
                 norm=None, activation=None, dw_conv=False, skip=False, mode=0, up_f=None):
        """
        
        Arguments:
            mode (int): 0--add;1--weighted;2--se;3--concat;
        """
        super().__init__()
        if max_level > 5: max_level = 5
        if bottom_up > max_level - min_level - 1: bottom_up = max_level - min_level - 1
        if bottom_up == 0: print('FPN style')
        if bottom_up and not skip: print('PANet style')
        if bottom_up and skip: print('BiFPN style')
        if mode == 0:
            print('Using addition')
        elif mode == 1:
            print('Using weighted addition')
        elif mode == 2:
            print('Using attentional addition')
        else:
            print('Using concat')
        self.bottom_up = bottom_up
        self.norm = norm
        self.up_f = [2 ** (i + 1) for i in range(5)] if up_f is None else up_f
        assert isinstance(self.up_f, list) and len(self.up_f) == 5
        self.skip = skip
        self.mode = mode

        if isinstance(dims, int):
            self.out_dim = dims
            fpn_dim = None
        else:
            fpn_dim, self.out_dim = dims[:2]
        self.num_stages = max_level - min_level

        self.top_down_ups = nn.ModuleList()
        self.top_down_nodes = nn.ModuleList()
        self.lateral_convs = nn.ModuleList() if fpn_dim is not None else None
        node_func = dcn_module if dcn else conv_module

        # build top down way, there are two ways: all layers have same channels(fpn_dim); each layer keep its channels
        for i in range(4, min_level, -1):
            temp_dim = num_filters[i - 1] if fpn_dim is None else fpn_dim
            in_dim = num_filters[i] if fpn_dim is None else fpn_dim
            if deconv_k:
                self.top_down_ups.append(deconv_module(in_dim, temp_dim, deconv_k, 2, norm, activation, dw_conv))
            else:
                if fpn_dim:
                    self.top_down_ups.append(nn.Upsample(scale_factor=2, mode='nearest'))
                else:
                    self.top_down_ups.append(
                        nn.Sequential(conv_module(num_filters[i], temp_dim, 1, 1, norm, activation),
                                      nn.Upsample(scale_factor=2, mode='nearest')))
            if mode == 3:
                self.top_down_nodes.append(node_func(2 * temp_dim, temp_dim, 3, 1, norm, activation, dw_conv))
            else:
                self.top_down_nodes.append(node_func(temp_dim, temp_dim, 3, 1, norm, activation, dw_conv))

        # while using the first way, converting all layers to have same channels first
        if self.lateral_convs is not None:
            for i in range(4, min_level - 1, -1):
                self.lateral_convs.append(conv_module(num_filters[i], fpn_dim, 1, 1, norm, activation))

        # build bottom up way, only use few layers for saving computation.
        if self.bottom_up:
            # keep index of the layers that will be used
            self.up_index = np.linspace(min_level, max_level - 1, bottom_up + 1, dtype=np.int)
            self.bottom_up_ups = nn.ModuleList()
            self.bottom_up_nodes = nn.ModuleList()
            self.bottom_up_convs = None if fpn_dim is not None and fpn_dim == self.out_dim else nn.ModuleList()
            for i, id in enumerate(self.up_index[1:]):
                temp_dim = num_filters[id] if fpn_dim is None else fpn_dim
                in_dim = temp_dim if self.bottom_up_convs is None else self.out_dim
                factor = self.up_f[id] / self.up_f[self.up_index[0]]
                if deconv_k:
                    self.bottom_up_ups.append(deconv_module(in_dim, self.out_dim, 3, factor, norm, activation, dw_conv))
                else:
                    if self.bottom_up_convs is not None:
                        self.bottom_up_ups.append(nn.Upsample(scale_factor=factor, mode='nearest'))
                    else:
                        self.bottom_up_ups.append(
                            nn.Sequential(conv_module(temp_dim, self.out_dim, 1, 1, norm, activation),
                                          nn.Upsample(scale_factor=factor, mode='nearest')))
                if mode == 3:
                    n = 3 if skip and not id == 4 else 2
                    self.bottom_up_nodes.append(
                        node_func(n * self.out_dim, self.out_dim, 3, 1, norm, activation, dw_conv))
                else:
                    self.bottom_up_nodes.append(node_func(self.out_dim, self.out_dim, 3, 1, norm, activation, dw_conv))

            if self.bottom_up_convs is not None:
                for id in self.up_index:
                    temp_dim = num_filters[id] if fpn_dim is None else fpn_dim
                    self.bottom_up_convs.append(conv_module(temp_dim, self.out_dim, 1, 1, norm, activation))

        if mode == 1:
            num1 = 5 - min_level - 1 if max_level < 5 else 5 - min_level
            num2 = len(self.up_index) - 1 if max_level < 5 else len(self.up_index) - 2
            n_w2 = 3 if skip else 2
            self.w1 = nn.Parameter(torch.rand(2, num1))
            self.w2 = nn.Parameter(torch.rand(n_w2, num2))
        elif mode == 2:
            self.se1 = nn.ModuleList()
            self.se2 = nn.ModuleList()
            for i in range(4, min_level, -1):
                temp_dim = num_filters[i - 1] if fpn_dim is None else fpn_dim
                self.se1.append(conv_module(2 * temp_dim, 2, 1, 1, norm, nn.Sigmoid))
            for id in self.up_index[1:]:
                if id == 4:
                    self.se2.append(conv_module(2 * self.out_dim, 2, 1, 1, norm, nn.Sigmoid))
                else:
                    if skip:
                        self.se2.append(conv_module(3 * self.out_dim, 3, 1, 1, norm, nn.Sigmoid))
                    else:
                        self.se2.append(conv_module(2 * self.out_dim, 2, 1, 1, norm, nn.Sigmoid))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                fill_up_weights(m)
            elif isinstance(m, nn.Conv2d):
                fill_fc_weights(m)
            elif self.norm is not None and isinstance(m, self.norm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        inputs = list(inputs)
        # first make all input layers have same channels if necessary
        if self.lateral_convs is not None:
            for i, l in enumerate(self.lateral_convs):
                inputs[-1 - i] = l(inputs[-1 - i])

        fpn_inner_blobs = [inputs[-1]]
        for i, node in enumerate(self.top_down_nodes):
            up = self.top_down_ups[i](fpn_inner_blobs[-1])
            if self.mode == 0:
                fusion = up + inputs[-2 - i]
            elif self.mode == 1:
                fusion = (self.w1[0, i] * inputs[-2 - i] + self.w1[1, i] * up) / (self.w1[0, i] + self.w1[1, i] + 1e-5)
            elif self.mode == 2:
                fusion_avg = torch.cat((F.adaptive_avg_pool2d(inputs[-2 - i], 1), F.adaptive_avg_pool2d(up, 1)), 1)
                fusion_w = self.se1[i](fusion_avg)
                fusion = (fusion_w[:, 0:1] * inputs[-2 - i] + fusion_w[:, 1:2] * up) / (
                        torch.sum(fusion_w, 1, keepdim=True) + 1e-5)
            else:
                fusion = torch.cat((inputs[-2 - i], up), 1)
            fpn_inner_blobs.append(node(fusion))

        fpn_output_blobs = []
        if self.bottom_up:
            fpn_output_blobs.append(fpn_inner_blobs[-1])
            if self.bottom_up_convs is not None:
                fpn_output_blobs[0] = self.bottom_up_convs[0](fpn_output_blobs[0])
            for i, id in enumerate(self.up_index[1:]):
                fpn_tmp = fpn_inner_blobs[-1 - id + self.up_index[0]]
                # convert channel first
                if self.bottom_up_convs is not None:
                    fpn_tmp = self.bottom_up_convs[i + 1](fpn_tmp)
                fpn_tmp = self.bottom_up_ups[i](fpn_tmp)
                # cases in which has three inputs
                if self.skip and not id == 4:
                    in_tmp = inputs[id]
                    if self.bottom_up_convs is not None:
                        in_tmp = self.bottom_up_convs[i + 1](in_tmp)
                    in_tmp = self.bottom_up_ups[i](in_tmp)
                if self.mode == 0:
                    if self.skip and not id == 4:
                        fusion = in_tmp + fpn_tmp + fpn_output_blobs[-1]
                    else:
                        fusion = fpn_tmp + fpn_output_blobs[-1]
                elif self.mode == 1:
                    if self.skip and not id == 4:
                        fusion = (self.w2[0, i] * in_tmp + self.w2[1, i] * fpn_tmp + self.w2[2, i] *
                                  fpn_output_blobs[-1]) / (self.w2[0, i] + self.w2[1, i] + self.w2[2, i] + 1e-5)
                    elif id == 4:
                        fusion = (self.w1[0, -1] * fpn_tmp + self.w1[1, -1] * fpn_output_blobs[-1]) / (
                                self.w1[0, -1] + self.w1[1, -1] + 1e-5)
                    else:
                        fusion = (self.w2[0, i] * fpn_tmp + self.w2[1, i] * fpn_output_blobs[-1]) / (
                                self.w2[0, i] + self.w2[1, i] + 1e-5)
                elif self.mode == 2:
                    if self.skip and not id == 4:
                        fusion_avg = torch.cat((F.adaptive_avg_pool2d(in_tmp, 1),
                                                F.adaptive_avg_pool2d(fpn_tmp, 1),
                                                F.adaptive_avg_pool2d(fpn_output_blobs[-1], 1)), 1)
                        fusion_w = self.se2[i](fusion_avg)
                        fusion = (fusion_w[:, 0:1] * in_tmp + fusion_w[:, 1:2] * fpn_tmp + fusion_w[:, 2:3] *
                                  fpn_output_blobs[-1]) / (torch.sum(fusion_w, 1, keepdim=True) + 1e-5)
                    else:
                        fusion_avg = torch.cat((F.adaptive_avg_pool2d(fpn_tmp, 1),
                                                F.adaptive_avg_pool2d(fpn_output_blobs[-1], 1)), 1)
                        fusion_w = self.se2[i](fusion_avg)
                        fusion = (fusion_w[:, 0:1] * fpn_tmp + fusion_w[:, 1:2] * fpn_output_blobs[-1]) / (
                                torch.sum(fusion_w, 1, keepdim=True) + 1e-5)
                else:
                    if self.skip and not id == 4:
                        fusion = torch.cat((in_tmp, fpn_tmp, fpn_output_blobs[-1]), 1)
                    else:
                        fusion = torch.cat((fpn_tmp, fpn_output_blobs[-1]), 1)
                fpn_tmp = self.bottom_up_nodes[i](fusion)
                fpn_output_blobs.append(fpn_tmp)
        else:
            fpn_output_blobs = fpn_inner_blobs

        return fpn_output_blobs[-1]


if __name__ == '__main__':
    x = [torch.rand(2, 16, 256, 256)]
    x.append(torch.rand(2, 24, 128, 128))
    x.append(torch.rand(2, 40, 64, 64))
    x.append(torch.rand(2, 112, 32, 32))
    x.append(torch.rand(2, 320, 16, 16))
    num_filters = [16, 24, 40, 112, 320]
    # neck = FPN(num_filters,256,bottom_up=True,deconv_k=4,norm=nn.BatchNorm2d,activation=nn.ReLU,dw_conv=True,max_level=7)
    # neck = StackBiFPN(num_filters,256,bottom_up=True,max_level=7,weighted=False,skip=False)
    neck = MdFPN(num_filters, 16, bottom_up=2, deconv_k=4, min_level=0, max_level=4, skip=True, mode=3)
    out = neck(x)
    for o in out:
        print(o.size())
