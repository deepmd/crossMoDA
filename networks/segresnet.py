from monai.networks.nets import SegResNet
from typing import Union, Optional, Tuple
from monai.utils import UpsampleMode

import torch
from torch import nn


class Projection(nn.Module):
    def __init__(self, head='mlp', dim_in=1024, feat_dim=128):
        super().__init__()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        x = self.head(x)
        x = nn.functional.normalize(x, dim=1)
        return x


class SegResNet(SegResNet):
    def __init__(
            self,
            spatial_dims: int = 3,
            init_filters: int = 8,
            in_channels: int = 1,
            out_channels: int = 2,
            dropout_prob: Optional[float] = None,
            norm_name: str = "group",
            num_groups: int = 8,
            use_conv_final: bool = True,
            blocks_down: tuple = (1, 2, 2, 4),
            blocks_up: tuple = (1, 1, 1),
            upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
            use_decoder: bool = True,
            head: str = 'mlp',
            feat_dim: int = 128,
    ):
        super().__init__(
            spatial_dims,
            init_filters,
            in_channels,
            out_channels,
            dropout_prob,
            norm_name,
            num_groups,
            use_conv_final,
            blocks_down,
            blocks_up,
            upsample_mode,
        )
        self.use_decoder = use_decoder
        if not self.use_decoder:
            self.up_layers = None
            self.up_samples = None
            input_projection_size = pow(2, len(self.down_layers) - 1) * self.init_filters
            self.projection = Projection(head=head, dim_in=input_projection_size, feat_dim=feat_dim)
        self.backbone = nn.Sequential(self.down_layers)

    def forward(self, x):
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        if not self.use_decoder:
            x = self.projection(x)
            return x

        down_x.reverse()

        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegResNet(spatial_dims=2,
                      init_filters=8,
                      in_channels=1,
                      out_channels=2,
                      blocks_down=(1, 2, 2, 4, 4, 4),
                      blocks_up=(1, 1, 1, 1, 1),
                      upsample_mode="deconv",
                      use_decoder=False,
                      dropout_prob=0.2,
                      head='mlp',
                      feat_dim=128,
                      ).to(device)

    x = torch.ones([3, 1, 384, 384]).to(device)
    out = model(x)
    print(out.size())
