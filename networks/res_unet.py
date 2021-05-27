from itertools import chain

import torch
import torch.nn as nn
from .projection import Projection


class PreActivateDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreActivateDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.double_conv(x)


class PreActivateResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreActivateResUpBlock, self).__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.double_conv = PreActivateDoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x) + self.ch_avg(x)


class PreActivateResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreActivateResBlock, self).__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))

        self.double_conv = PreActivateDoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.ch_avg(x)
        out = self.double_conv(x)
        out = out + identity
        return self.down_sample(out), out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)


class ResUNet(nn.Module):
    """
        Based on "Road Extraction by Deep Residual U-Net"
        https://arxiv.org/abs/1711.10684
        https://github.com/galprz/brain-tumor-segmentation
    """
    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            filters=(64, 128, 256, 512),
            use_decoder=True,
            head='mlp',
            feat_dim=128,
            use_conv_final=True,
    ):
        super(ResUNet, self).__init__()
        self.use_conv_final = use_conv_final
        self.use_decoder = use_decoder
        if not self.use_decoder:
            self.projection = Projection(head=head, dim_in=filters[-1], feat_dim=feat_dim)

        self.down_conv1 = PreActivateResBlock(in_channels, filters[0])
        self.down_conv2 = PreActivateResBlock(filters[0], filters[1])
        self.down_conv3 = PreActivateResBlock(filters[1], filters[2])
        self.down_conv4 = PreActivateResBlock(filters[2], filters[3])

        self.double_conv = PreActivateDoubleConv(filters[3], filters[4])

        if self.use_decoder:
            self.up_conv4 = PreActivateResUpBlock(filters[3] + filters[4], filters[3])
            self.up_conv3 = PreActivateResUpBlock(filters[2] + filters[3], filters[2])
            self.up_conv2 = PreActivateResUpBlock(filters[1] + filters[2], filters[1])
            self.up_conv1 = PreActivateResUpBlock(filters[1] + filters[0], filters[0])

            if self.use_conv_final:
                self.conv_last = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def freeze_encoder(self):
        encoder_layers = [self.down_conv1, self.down_conv2, self.down_conv3, self.down_conv4, self.double_conv]
        for param in chain.from_iterable(layer.parameters() for layer in encoder_layers):
            param.requires_grad = False

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)

        if not self.use_decoder:
            x = self.projection(x)
            return x

        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)

        if self.use_conv_final:
            x = self.conv_last(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResUNet(in_channels=1,
                    out_channels=2,
                    filters=(64, 128, 256, 512, 1024),
                    use_decoder=True,
                    head='mlp',
                    feat_dim=128,
                    use_conv_final=True
                    ).to(device)

    x = torch.ones([3, 1, 352, 352]).to(device)
    out = model(x)
    print(out.size())
