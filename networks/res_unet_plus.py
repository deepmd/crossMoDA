import torch.nn as nn
import torch
from .modules import (
    ResidualConv,
    ASPP,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
    Projection,
)


class ResUnetPlusPlus(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            filters=(32, 64, 128, 256, 512),
            use_decoder=True,
            head='mlp',
            feat_dim=128
    ):
        super(ResUnetPlusPlus, self).__init__()

        self.use_decoder = use_decoder

        if not self.use_decoder:
            self.projection = Projection(head=head, dim_in=filters[-1], feat_dim=feat_dim)

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

        if self.use_decoder:

            self.attn1 = AttentionBlock(filters[3], filters[5], filters[5])
            self.upsample1 = Upsample_(2)
            self.up_residual_conv1 = ResidualConv(filters[5] + filters[3], filters[4], 1, 1)

            self.attn2 = AttentionBlock(filters[2], filters[4], filters[4])
            self.upsample2 = Upsample_(2)
            self.up_residual_conv2 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

            self.attn3 = AttentionBlock(filters[1], filters[3], filters[3])
            self.upsample3 = Upsample_(2)
            self.up_residual_conv3 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

            self.attn4 = AttentionBlock(filters[0], filters[2], filters[2])
            self.upsample4 = Upsample_(2)
            self.up_residual_conv4 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

            self.aspp_out = ASPP(filters[1], filters[0])

            self.output_layer = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.squeeze_excite4(x4)
        x5 = self.residual_conv4(x5)

        x6 = self.aspp_bridge(x5)

        if not self.use_decoder:
            x = self.projection(x6)
            return x

        x7 = self.attn1(x4, x6)
        x7 = self.upsample1(x7)
        x7 = torch.cat([x7, x4], dim=1)
        x7 = self.up_residual_conv1(x7)

        x8 = self.attn2(x3, x7)
        x8 = self.upsample2(x8)
        x8 = torch.cat([x8, x3], dim=1)
        x8 = self.up_residual_conv2(x8)

        x9 = self.attn3(x2, x8)
        x9 = self.upsample3(x9)
        x9 = torch.cat([x9, x2], dim=1)
        x9 = self.up_residual_conv3(x9)

        x10 = self.attn4(x1, x9)
        x10 = self.upsample4(x10)
        x10 = torch.cat([x10, x1], dim=1)
        x10 = self.up_residual_conv4(x10)

        x11 = self.aspp_out(x10)
        out = self.output_layer(x11)

        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResUnetPlusPlus(in_channels=1,
                            out_channels=2,
                            filters=(64, 128, 256, 512, 1024, 1024),
                            use_decoder=False,
                            head='mlp',
                            feat_dim=128,
                            ).to(device)

    x = torch.ones([3, 1, 352, 352]).to(device)
    out = model(x)
    print(out.size())

