import torch
import torch.nn as nn
from modules import ResidualConv, Upsample, Projection


class ResUnet(nn.Module):
    def __init__(
            self,
            channel,
            filters=[64, 128, 256, 512],
            use_decoder=True,
            head='mlp',
            feat_dim=128):
        super(ResUnet, self).__init__()

        self.use_decoder = use_decoder

        if not self.use_decoder:
            self.projection = Projection(head=head, dim_in=filters[-1], feat_dim=feat_dim)

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.bridge = ResidualConv(filters[3], filters[4], 2, 1)

        if self.use_decoder:
            self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
            self.up_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)

            self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
            self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

            self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
            self.up_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

            self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
            self.up_residual_conv4 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

            self.output_layer = nn.Sequential(
                nn.Conv2d(filters[1], 1, 1, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        # Bridge
        x5 = self.bridge(x4)

        if not self.use_decoder:
            x = self.projection(x5)
            return x

        # Decode
        x5 = self.upsample_1(x5)
        x6 = torch.cat([x5, x4], dim=1)

        x7 = self.up_residual_conv1(x6)

        x7 = self.upsample_2(x7)
        x8 = torch.cat([x7, x3], dim=1)

        x9 = self.up_residual_conv2(x8)

        x9 = self.upsample_3(x9)
        x10 = torch.cat([x9, x2], dim=1)

        x11 = self.up_residual_conv3(x10)

        output = self.output_layer(x11)

        return output


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResUnet(channel=1,
                    filters=[64, 128, 256, 512, 1024],
                    use_decoder=False,
                    head='mlp',
                    feat_dim=128,
                    ).to(device)

    x = torch.ones([3, 1, 352, 352]).to(device)
    out = model(x)
    print(out.size())
