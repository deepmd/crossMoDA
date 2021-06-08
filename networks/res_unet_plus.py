from itertools import chain

import torch.nn as nn
import torch


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample(nn.Module):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale, align_corners=True)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2

class ResUNetPlusPlus(nn.Module):
    """
        Based on "ResUNet++: An Advanced Architecture for Medical Image Segmentation"
        https://arxiv.org/abs/1911.07067
        https://github.com/rishikksh20/ResUnet
    """
    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            filters=(32, 64, 128, 256, 512),
            use_decoder={"projector": False, "index_decoder_block": -1, "output": False},
            use_conv_final=True
    ):
        super(ResUNetPlusPlus, self).__init__()
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

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.squeeze_excite4 = Squeeze_Excite_Block(filters[3])
        self.residual_conv4 = ResidualConv(filters[3], filters[4], 2, 1)

        self.aspp_bridge = ASPP(filters[4], filters[5])

        if self.use_decoder_projector or self.use_decoder_output:
            self.attn1 = AttentionBlock(filters[3], filters[5], filters[5])
            self.upsample1 = Upsample(2)
            self.up_residual_conv1 = ResidualConv(filters[5] + filters[3], filters[4], 1, 1)

            self.attn2 = AttentionBlock(filters[2], filters[4], filters[4])
            self.upsample2 = Upsample(2)
            self.up_residual_conv2 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

            self.attn3 = AttentionBlock(filters[1], filters[3], filters[3])
            self.upsample3 = Upsample(2)
            self.up_residual_conv3 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

            self.attn4 = AttentionBlock(filters[0], filters[2], filters[2])
            self.upsample4 = Upsample(2)
            self.up_residual_conv4 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

            self.aspp_out = ASPP(filters[1], filters[0])

            if self.use_conv_final:
                self.conv_last = nn.Conv2d(filters[0], out_channels, 1)

    def freeze_encoder(self):
        encoder_layers = [self.input_layer, self.input_skip, self.squeeze_excite1, self.residual_conv1,
                          self.squeeze_excite2, self.residual_conv2, self.squeeze_excite3, self.residual_conv3,
                          self.squeeze_excite4, self.residual_conv4, self.aspp_bridge]
        for param in chain.from_iterable(layer.parameters() for layer in encoder_layers):
            param.requires_grad = False

    def get_encoder_output_size(self, image_size):
        height, width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        # Note that first layer and Bridge layer don't change dimensions of inputs.
        num_downs_in_encoder = len(self.filters) - 2
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
        num_ups_until_model_output = 0 if index_decoder_block in [-1, -2] else abs(2+index_decoder_block)
        height_dec_feat = height // 2 ** num_ups_until_model_output
        width_dec_feat = width // 2 ** num_ups_until_model_output

        # Reverse the encoder filters list to gain decoder filters list.
        up_filters = self.filters[::-1]
        channels_dec_feat = up_filters[index_decoder_block]
        return {'channels': channels_dec_feat, 'height': height_dec_feat, 'width': width_dec_feat}

    def forward(self, x, show_size=False):
        x1 = self.input_layer(x) + self.input_skip(x)
        if show_size: print('x1=', x1.size())

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)
        if show_size: print('x2=', x2.size())

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)
        if show_size: print('x3=', x3.size())

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)
        if show_size: print('x4=', x4.size())

        x5 = self.squeeze_excite4(x4)
        x5 = self.residual_conv4(x5)
        if show_size: print('x5=', x5.size())

        x6 = self.aspp_bridge(x5)
        if show_size: print('x6=', x6.size())

        if self.use_decoder_projector or self.use_decoder_output:
            x7 = self.attn1(x4, x6)
            x7 = self.upsample1(x7)
            x7 = torch.cat([x7, x4], dim=1)
            x7 = self.up_residual_conv1(x7)
            if show_size: print('x7=', x7.size())
            if self.use_decoder_projector and self.index == -5:
                return x7

            x8 = self.attn2(x3, x7)
            x8 = self.upsample2(x8)
            x8 = torch.cat([x8, x3], dim=1)
            x8 = self.up_residual_conv2(x8)
            if show_size: print('x8=', x8.size())
            if self.use_decoder_projector and self.index == -4:
                return x8

            x9 = self.attn3(x2, x8)
            x9 = self.upsample3(x9)
            x9 = torch.cat([x9, x2], dim=1)
            x9 = self.up_residual_conv3(x9)
            if show_size: print('x9=', x9.size())
            if self.use_decoder_projector and self.index == -3:
                return x9

            x10 = self.attn4(x1, x9)
            x10 = self.upsample4(x10)
            x10 = torch.cat([x10, x1], dim=1)
            x10 = self.up_residual_conv4(x10)
            if show_size: print('x10=', x10.size())
            if self.use_decoder_projector and self.index == -2:
                return x10

            x11 = self.aspp_out(x10)
            if show_size: print('x11=', x11.size())
            if self.use_decoder_projector and self.index == -1:
                return x11

            if self.use_conv_final:
                x11 = self.conv_last(x11)

        return x11 if self.use_decoder_output else x6


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResUNetPlusPlus(in_channels=1,
                            out_channels=2,
                            filters=(64, 128, 256, 512, 1024, 1024),
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

