import torch
from torch import nn
from modules import Projection
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
    def __init__(self, inplanes, in_filters):
        super().__init__()
        expansion = 4
        F1, F2, F3 = in_filters, in_filters, in_filters * expansion
        self.conv_shortcut = conv1x1(inplanes, F3, stride=1)
        self.conv1 = conv1x1(inplanes, F1, stride=1)
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
            inplanes,
            planes,
            downsample=False,
    ):
        super().__init__()
        expansion = 4
        F1, F2, F3 = planes, planes, planes * expansion
        self.downsample = downsample
        stride = 2 if self.downsample else 1

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv1x1(inplanes, F1, stride=stride)
        self.bn2 = nn.BatchNorm2d(F1)
        self.conv2 = conv3x3(F1, F2, stride=1)
        self.bn3 = nn.BatchNorm2d(F2)
        self.conv3 = conv1x1(F2, F3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        if self.downsample:
            self.conv_shortcut = conv1x1(inplanes, F3, stride=stride)

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

    def forward(self, x_down, x_skip):
        x_down = self.upsample(x_down)
        x = torch.cat([x_down, x_skip], dim=1)

        return x


class DR_Unet104(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 init_filters,
                 layers,
                 dropout=0,
                 use_decoder=False,
                 head='mlp',
                 feat_dim=128,
                 use_conv_final=True
                 ):
        super().__init__()
        self.use_conv_final = use_conv_final
        self.use_decoder = use_decoder
        if not self.use_decoder:
            self.projection = Projection(head=head, dim_in=2048, feat_dim=feat_dim)
        filters = [init_filters * 2 ** i for i in range(len(layers))]
        # inplanes = [init_filters * 2 ** i for i in range(2, len(layers)+1)]
        self.input_bottleneck_block = InputBottleneckBlock(in_channels, filters[0])
        self.bottleneck_block = BottleneckBlock(filters[0]*4, filters[0])

        self.down_layer1 = self._make_down_layer(filters[1]*2, filters[1], layers[1], dropout)
        self.down_layer2 = self._make_down_layer(filters[2]*2, filters[2], layers[2], dropout)
        self.down_layer3 = self._make_down_layer(filters[3]*2, filters[3], layers[3], dropout)
        self.down_layer4 = self._make_down_layer(filters[4]*2, filters[4], layers[4], dropout)
        self.bridge = self._make_down_layer(filters[5]*2, filters[5], layers[5], dropout)

        if self.use_decoder:
            # Level 5
            self.upsample_and_concatenation1 = UpsampleConcatenation(2048, 1024)
            self.residual_block1 = ResidualBlock(1024+1024, 512)

            # Level 4
            self.upsample_and_concatenation2 = UpsampleConcatenation(512, 512)
            self.residual_block2 = ResidualBlock(512+512, 256)

            # Level 3
            self.upsample_and_concatenation3 = UpsampleConcatenation(256, 256)
            self.residual_block3 = ResidualBlock(256+256, 128)

            # Level 2
            self.upsample_and_concatenation4 = UpsampleConcatenation(128, 128)
            self.residual_block4 = ResidualBlock(128+128, 64)

            # Level 1
            self.upsample_and_concatenation5 = UpsampleConcatenation(64, 64)
            self.residual_block5 = ResidualBlock(64+64, 32)
            self.bn = nn.BatchNorm2d(32)
            self.relu = nn.ReLU(inplace=True)

            # Output layer
            if self.use_conv_final:
                self.conv_final = conv1x1(32, out_channels)

    def _make_down_layer(self, inplanes, planes, blocks, dropout):
        layers = list()
        layers.append(BottleneckBlock(inplanes, planes, downsample=True))
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(planes*4, planes, downsample=False))

        if dropout != 0:
            layers.append(nn.Dropout(p=dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Level 1
        e0 = self.input_bottleneck_block(x)
        e0 = self.bottleneck_block(e0)
        print(e0.size())

        # Level 2
        e1 = self.down_layer1(e0)
        print(e1.size())

        # Level 3
        e2 = self.down_layer2(e1)
        print(e2.size())

        # Level 4
        e3 = self.down_layer3(e2)
        # x4 = self.zero_padding(x4)
        print(e3.size())

        # Level 5
        e4 = self.down_layer4(e3)
        print(e4.size())

        # Bridge
        e5 = self.bridge(e4)
        print(e5.size())

        if not self.use_decoder:
            x = self.projection(e5)
            return x

        # Level 5
        u0 = self.upsample_and_concatenation1(e5, e4)
        d0 = self.residual_block1(u0)
        print(d0.size())

        # Level 4
        u1 = self.upsample_and_concatenation2(d0, e3)
        d1 = self.residual_block2(u1)
        print(d1.size())

        # Level 3
        u2 = self.upsample_and_concatenation3(d1, e2)
        d2 = self.residual_block3(u2)
        print(d2.size())

        # Level 2
        u3 = self.upsample_and_concatenation4(d2, e1)
        d3 = self.residual_block4(u3)
        print(d3.size())

        # Level 1
        u4 = self.upsample_and_concatenation5(d3, e0)
        d4 = self.residual_block5(u4)
        x = self.bn(d4)
        x = self.relu(x)
        print(x.size())

        # Output layer
        if self.use_conv_final:
            x = self.conv_final(x)

        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DR_Unet104(in_channels=1,
                       out_channels=2,
                       init_filters=16,
                       layers=[2, 3, 3, 5, 14, 4],
                       dropout=0.2,
                       use_decoder=False,
                       head='mlp',
                       feat_dim=128,
                       ).to(device)

    x = torch.ones([3, 1, 352, 352]).to(device)
    out = model(x)
    print('out=', out.size())
