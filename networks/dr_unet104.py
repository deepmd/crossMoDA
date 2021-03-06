from itertools import chain

import torch
from torch import nn
from typing import Union, Sequence, Tuple
import numpy as np


def same_padding(
    kernel_size: Union[Sequence[int], int], dilation: Union[Sequence[int], int] = 1
) -> Union[Tuple[int, ...], int]:
    """
    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.

    Raises:
        NotImplementedError: When ``np.any((kernel_size - 1) * dilation % 2 == 1)``.

    """

    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)

    if np.any((kernel_size_np - 1) * dilation % 2 == 1):
        raise NotImplementedError(
            f"Same padding not available for kernel_size={kernel_size_np} and dilation={dilation_np}."
        )

    padding_np = (kernel_size_np - 1) / 2 * dilation_np
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def stride_minus_kernel_padding(
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)

    out_padding_np = stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=None):
    """3x3 convolution with padding"""
    if padding is None:
        padding = same_padding(kernel_size=3, dilation=dilation)

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


def conv1x1(in_planes, out_planes, stride=1, dilation=1, padding=None):
    """1x1 convolution"""
    if padding is None:
        padding = same_padding(kernel_size=1, dilation=dilation)

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=False)


class InputBottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        F1, F2, F3 = out_planes//4, out_planes//4, out_planes
        self.conv_shortcut = conv1x1(in_planes, F3, stride=1)
        self.conv1 = conv1x1(in_planes, F1, stride=1)
        self.bn2 = nn.BatchNorm2d(F1)
        self.conv2 = conv3x3(F1, F2, stride=1)
        self.bn3 = nn.BatchNorm2d(F2)
        self.conv3 = conv1x1(F2, F3, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_shortcut = self.conv_shortcut(x)

        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)

        x += x_shortcut

        return x


class BottleneckBlock(nn.Module):
    def __init__(
            self,
            in_planes,
            out_planes,
            downsample=False,
    ):
        super().__init__()
        F1, F2, F3 = out_planes//4, out_planes//4, out_planes
        self.downsample = downsample
        stride = 2 if self.downsample else 1

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv1x1(in_planes, F1, stride=stride)
        self.bn2 = nn.BatchNorm2d(F1)
        self.conv2 = conv3x3(F1, F2, stride=1)
        self.bn3 = nn.BatchNorm2d(F2)
        self.conv3 = conv1x1(F2, F3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        if self.downsample:
            self.conv_shortcut = conv1x1(in_planes, F3, stride=stride)

    def forward(self, x):
        x_shortcut = x

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)

        if self.downsample:
            x_shortcut = self.conv_shortcut(x_shortcut)

        x += x_shortcut

        return x


class ResidualBlock(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
    ):
        super().__init__()
        self.conv_shortcut = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_shortcut = self.conv_shortcut(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x += x_shortcut

        return x


class UpsampleConcatenation(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            padding=None,
            output_padding=None,
            kernel_size=3,
            dilation=1,
            stride=2
    ):
        super().__init__()
        if padding is None:
            padding = same_padding(kernel_size=kernel_size, dilation=dilation)

        if output_padding is None:
            output_padding = stride_minus_kernel_padding(1, stride=stride)

        self.upsample = nn.ConvTranspose2d(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding)

    def forward(self, x):
        x_down, x_skip = x
        x_down = self.upsample(x_down)
        x = torch.cat([x_down, x_skip], dim=1)

        return x


class DRUNet104(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 init_filters,
                 layers,
                 dropout=0,
                 use_decoder={"projector": False, "index_decoder_block": -1, "output": False},
                 use_conv_final=True
                 ):
        super().__init__()
        self.use_conv_final = use_conv_final
        self.use_decoder_projector = use_decoder["projector"]
        self.use_decoder_output = use_decoder["output"]
        self.index = use_decoder["index_decoder_block"]

        # for init_filters=16: filters = [64, 128, 256, 512, 1024, 2048]
        self.filters = [init_filters * 4 * 2 ** i for i in range(len(layers))]

        num_ups_in_decoder = len(self.filters) - 1
        if self.use_decoder_projector:
            if -self.index > num_ups_in_decoder or self.index >= 0:
                raise ValueError(f"Decoder has only {num_ups_in_decoder} blocks."
                                 f" [-1, -{num_ups_in_decoder}] indices are valid.")

        self.input_bottleneck_block = InputBottleneckBlock(in_channels, self.filters[0])
        self.bottleneck_block = BottleneckBlock(self.filters[0], self.filters[0])

        self.down_layer1 = self._make_down_layer(self.filters[0], self.filters[1], layers[1], dropout)
        self.down_layer2 = self._make_down_layer(self.filters[1], self.filters[2], layers[2], dropout)
        self.down_layer3 = self._make_down_layer(self.filters[2], self.filters[3], layers[3], dropout)
        self.down_layer4 = self._make_down_layer(self.filters[3], self.filters[4], layers[4], dropout)
        self.bridge = self._make_down_layer(self.filters[4], self.filters[5], layers[5], dropout)

        if self.use_decoder_projector or self.use_decoder_output:
            self.up_layer1 = self._make_up_layer(self.filters[5], self.filters[4]//2, dropout)
            self.up_layer2 = self._make_up_layer(self.filters[4]//2, self.filters[3]//2, dropout)
            self.up_layer3 = self._make_up_layer(self.filters[3]//2, self.filters[2]//2, dropout)
            self.up_layer4 = self._make_up_layer(self.filters[2]//2, self.filters[1]//2, dropout)
            self.up_layer5 = self._make_up_layer(self.filters[1]//2, self.filters[0]//2, dropout)
            self.bn = nn.BatchNorm2d(self.filters[0]//2)
            self.relu = nn.ReLU(inplace=True)

            # Output layer
            if self.use_conv_final:
                self.conv_final = conv1x1(self.filters[0]//2, out_channels)

    def _make_up_layer(self, in_planes, out_planes, dropout):
        layers = list()
        layers.append(UpsampleConcatenation(in_planes, out_planes*2))
        layers.append(ResidualBlock(out_planes*4, out_planes))

        if dropout != 0:
            layers.append(nn.Dropout(p=dropout))

        return nn.Sequential(*layers)

    def _make_down_layer(self, in_planes, out_planes, blocks, dropout):
        layers = list()
        layers.append(BottleneckBlock(in_planes, out_planes, downsample=True))
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(out_planes, out_planes, downsample=False))

        if dropout != 0:
            layers.append(nn.Dropout(p=dropout))

        return nn.Sequential(*layers)

    def freeze_encoder(self):
        encoder_layers = [self.input_bottleneck_block, self.bottleneck_block, self.down_layer1, self.down_layer2,
                          self.down_layer3, self.down_layer4, self.bridge]
        for param in chain.from_iterable(layer.parameters() for layer in encoder_layers):
            param.requires_grad = False

    def get_encoder_output_size(self, image_size):
        height, width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        # Note that Bridge layer doesn't change dimensions of inputs.
        num_downs_in_encoder = len(self.filters) - 1
        height_en_feat = height // 2 ** num_downs_in_encoder
        width_en_feat = width // 2 ** num_downs_in_encoder
        channels_en_feat = self.filters[-1]
        return {'channels': channels_en_feat, 'height': height_en_feat, 'width': width_en_feat}

    def get_decoder_output_size(self, image_size, index_decoder_block):
        height, width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        num_ups_in_decoder = len(self.filters) - 1
        if -index_decoder_block > num_ups_in_decoder or index_decoder_block >= 0:
            raise ValueError(f"Decoder has only {num_ups_in_decoder} blocks."
                             f" Expected range of indices is [-1, -{num_ups_in_decoder}].")

        # How many upsample occure from index_decoder_block until end of decoder
        num_ups_until_model_output = abs(1+index_decoder_block)
        height_dec_feat = height // 2 ** num_ups_until_model_output
        width_dec_feat = width // 2 ** num_ups_until_model_output

        # Reverse the encoder filters list to gain decoder filters list.
        up_filters = self.filters[::-1]
        channels_dec_feat = up_filters[index_decoder_block] // 2
        return {'channels': channels_dec_feat, 'height': height_dec_feat, 'width': width_dec_feat}

    def forward(self, x, show_size=False):
        # Level 1
        e0 = self.input_bottleneck_block(x)
        e0 = self.bottleneck_block(e0)
        if show_size: print('e0=', e0.size())
        # Level 2
        e1 = self.down_layer1(e0)
        if show_size: print('e1=', e1.size())
        # Level 3
        e2 = self.down_layer2(e1)
        if show_size: print('e2=', e2.size())
        # Level 4
        e3 = self.down_layer3(e2)
        if show_size: print('e3=', e3.size())
        # Level 5
        e4 = self.down_layer4(e3)
        if show_size: print('e4=', e4.size())
        # Bridge
        e5 = self.bridge(e4)
        if show_size: print('e5=', e5.size())

        if self.use_decoder_projector or self.use_decoder_output:
            # Level 5
            d0 = self.up_layer1([e5, e4])
            if show_size: print('d0=', d0.size())
            if self.use_decoder_projector and self.index == -5:
                return d0
            # Level 4
            d1 = self.up_layer2([d0, e3])
            if show_size: print('d1=', d1.size())
            if self.use_decoder_projector and self.index == -4:
                return d1
            # Level 3
            d2 = self.up_layer3([d1, e2])
            if show_size: print('d2=', d2.size())
            if self.use_decoder_projector and self.index == -3:
                return d2
            # Level 2
            d3 = self.up_layer4([d2, e1])
            if show_size: print('d3=', d3.size())
            if self.use_decoder_projector and self.index == -2:
                return d3
            # Level 1
            d4 = self.up_layer5([d3, e0])
            x = self.bn(d4)
            x = self.relu(x)
            if show_size: print('d4=', d4.size())
            if self.use_decoder_projector and self.index == -1:
                return x

            # Output layer
            if self.use_conv_final:
                x = self.conv_final(x)

        return x if self.use_decoder_output else e5


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DRUNet104(in_channels=1,
                      out_channels=2,
                      init_filters=16,
                      layers=[2, 3, 3, 5, 14, 4],
                      dropout=0.2,
                      use_decoder={"projector": False, "index_decoder_block": -1, "output": False},
                      use_conv_final=False,
                      ).to(device)

    en_size = model.get_encoder_output_size(image_size=352)
    print('encoder_output_size=', en_size)
    dec_size = model.get_decoder_output_size(image_size=352, index_decoder_block=-1)
    print('decoder_output_size=', dec_size)

    x = torch.ones([3, 1, 352, 352]).to(device)
    out = model(x, show_size=True)
    print('out=', out.size())
