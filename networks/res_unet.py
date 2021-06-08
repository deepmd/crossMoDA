from itertools import chain

import torch
import torch.nn as nn


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
            use_decoder={"projector": False, "index_decoder_block": -1, "output": False},
            use_conv_final=True,
    ):
        super(ResUNet, self).__init__()
        self.filters = filters
        self.use_conv_final = use_conv_final
        self.use_decoder_projector = use_decoder["projector"]
        self.use_decoder_output = use_decoder["output"]
        self.index = use_decoder["index_decoder_block"]

        num_ups_in_decoder = len(filters) - 1
        if self.use_decoder_projector:
            if -self.index > num_ups_in_decoder or self.index >= 0:
                raise ValueError(f"Decoder has only {num_ups_in_decoder} blocks."
                                 f" [-1, -{num_ups_in_decoder}] indices are valid.")

        self.down_conv1 = PreActivateResBlock(in_channels, filters[0])
        self.down_conv2 = PreActivateResBlock(filters[0], filters[1])
        self.down_conv3 = PreActivateResBlock(filters[1], filters[2])
        self.down_conv4 = PreActivateResBlock(filters[2], filters[3])

        self.double_conv = PreActivateDoubleConv(filters[3], filters[4])

        if self.use_decoder_projector or self.use_decoder_output:
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
                             f" [-1, -{num_ups_in_decoder}] indices are valid.")

        # How many upsample occure from index_decoder_block until end of decoder
        num_ups_until_model_output = abs(1+index_decoder_block)
        height_dec_feat = height // 2 ** num_ups_until_model_output
        width_dec_feat = width // 2 ** num_ups_until_model_output

        # Reverse the encoder filters list to gain decoder filters list.
        up_filters = self.filters[::-1]
        channels_dec_feat = up_filters[index_decoder_block]
        return {'channels': channels_dec_feat, 'height': height_dec_feat, 'width': width_dec_feat}

    def forward(self, x, show_size=False):
        x, skip1_out = self.down_conv1(x)
        if show_size: print(x.size())
        x, skip2_out = self.down_conv2(x)
        if show_size: print(x.size())
        x, skip3_out = self.down_conv3(x)
        if show_size: print(x.size())
        x, skip4_out = self.down_conv4(x)
        if show_size: print(x.size())
        x = self.double_conv(x)
        if show_size: print(x.size())

        if self.use_decoder_projector or self.use_decoder_output:
            x = self.up_conv4(x, skip4_out)
            if show_size: print(x.size())
            if self.use_decoder_projector and self.index == -4:
                return x

            x = self.up_conv3(x, skip3_out)
            if show_size: print(x.size())
            if self.use_decoder_projector and self.index == -3:
                return x

            x = self.up_conv2(x, skip2_out)
            if show_size: print(x.size())
            if self.use_decoder_projector and self.index == -2:
                return x

            x = self.up_conv1(x, skip1_out)
            if show_size: print(x.size())
            if self.use_decoder_projector and self.index == -1:
                return x

            if self.use_conv_final:
                x = self.conv_last(x)

        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResUNet(in_channels=1,
                    out_channels=2,
                    filters=(64, 128, 256, 512, 1024),
                    use_decoder={"projector": False, "index_decoder_block": -1, "output": False},
                    use_conv_final=True
                    ).to(device)
    en_size = model.get_encoder_output_size(image_size=352)
    print('encoder_output_size=', en_size)
    dec_size = model.get_decoder_output_size(image_size=352, index_decoder_block=-4)
    print('decoder_output_size=', dec_size)

    x = torch.ones([3, 1, 352, 352]).to(device)
    out = model(x, show_size=True)
    print('out=', out.size())
