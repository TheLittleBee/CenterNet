from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch.nn import functional as F
import math

from .utils4ENet import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)
from .fpn import MdFPN, fill_fc_weights

BN = nn.BatchNorm2d
# BN = nn.SyncBatchNorm
# BN = nn.GroupNorm


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        ex = x
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x
        # se = x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x, ex

    def set_swish(self, memory_efficient=True, relu=False):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        if relu: self._swish = nn.ReLU(inplace=True)
        else: self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self._swish = MemoryEfficientSwish()
        self.ex = False

    def set_swish(self, memory_efficient=True, relu=False):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        if relu: self._swish = nn.ReLU(inplace=True)
        else: self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient, relu)

    def forward(self, inputs):
        """ Returns output of the final convolution layer """
        # get multi features for FPN
        features = [] if hasattr(self, 'num_blocks') else None
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        if features is not None and 0 in getattr(self, 'num_blocks'): features.append(x)
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x, ex = block(x, drop_connect_rate=drop_connect_rate)
            if features is not None and idx + 1 in getattr(self, 'num_blocks'):
                if self.ex: features.append(ex)
                else: features.append(x)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        if features is not None and len(self._blocks) + 1 in getattr(self, 'num_blocks'): features.append(x)
        if features is not None: return features
        return x

    def get_filters(self, num_blocks=(1, 2, 3, 5, 8), ex=False):
        """
        get channels of features, set layers where features are
        """
        self.ex = ex
        num_filters = []
        blocks = []
        n_layer = 0
        if 0 in num_blocks:
            blocks.append(n_layer)
            num_filters.append(round_filters(32, self._global_params))
        for i, block_args in enumerate(self._blocks_args):
            rp = round_repeats(block_args.num_repeat, self._global_params)
            n_layer += rp
            if i + 1 in num_blocks:
                blocks.append(n_layer)
                if ex:
                    filters = block_args.input_filters if rp == 1 else block_args.output_filters
                    num_filters.append(round_filters(filters * block_args.expand_ratio, self._global_params))
                else: num_filters.append(round_filters(block_args.output_filters, self._global_params))
        if 8 in num_blocks:
            blocks.append(n_layer + 1)
            num_filters.append(round_filters(1280, self._global_params))
        setattr(self, 'num_blocks', blocks)
        return num_filters

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name)

        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet-b' + str(i) for i in range(num_models)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


class PoseEfficientNet(nn.Module):
    def __init__(self, name, heads, head_conv, down_ratio=4, max_level=5, dcn=False, bottom_up=2, deconv_k=4,
                 activation=None, dw_conv=False, skip=False, mode=3):
        super().__init__()
        self.heads = heads
        min_level = int(math.log2(down_ratio)) - 1
        self.backbone = EfficientNet.from_pretrained(name)
        if activation is not None: self.backbone.set_swish(relu=True)
        activation = MemoryEfficientSwish if activation is None else activation
        num_filters = self.backbone.get_filters()
        out_dim = num_filters[0]
        # out_dim = 32
        self.fpn = MdFPN(num_filters, out_dim, min_level, max_level, dcn, bottom_up, deconv_k, BN, activation,
                         dw_conv, skip, mode)

        # self.conv = nn.Sequential(nn.Conv2d(64,out_dim,1,bias=False),
        #                           BN(out_dim),
        #                           activation())
        head_conv = out_dim
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(out_dim, head_conv,
                              kernel_size=3, padding=1, bias=True),
                    activation(),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True))
                if 'hm' in head or 'obj' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(out_dim, classes,
                               kernel_size=1, stride=1,
                               padding=0, bias=True)
                if 'hm' in head or 'obj' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        features = self.backbone(x)
        x = self.fpn(features)
        # x = self.conv(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


def get_pose_net(num_layers, heads, head_conv=16, down_ratio=4, **kwargs):
    name = 'efficientnet-b%d' % num_layers
    model = PoseEfficientNet(name, heads, head_conv=head_conv, down_ratio=down_ratio, **kwargs)
    return model
