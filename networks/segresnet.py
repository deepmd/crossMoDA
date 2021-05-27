from itertools import chain

from monai.networks.nets import SegResNet as monaiSegResNet
from typing import Union, Optional
from monai.utils import UpsampleMode
from .projection import Projection

import torch


class SegResNet(monaiSegResNet):
    """
        Based on "3D MRI brain tumor segmentation using autoencoder regularization"
        https://arxiv.org/abs/1810.11654
        The module does not include the variational autoencoder (VAE).
    """
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
            self.conv_final = None
            input_projection_size = pow(2, len(self.down_layers) - 1) * self.init_filters
            self.projection = Projection(head=head, dim_in=input_projection_size, feat_dim=feat_dim)

    def freeze_encoder(self):
        encoder_layers = [self.convInit] + self.down_layers
        for param in chain.from_iterable(layer.parameters() for layer in encoder_layers):
            param.requires_grad = False

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
                      init_filters=32,
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
