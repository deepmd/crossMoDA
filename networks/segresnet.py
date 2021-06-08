from itertools import chain

from monai.networks.nets import SegResNet as monaiSegResNet
from typing import Union, Optional, Dict
from monai.utils import UpsampleMode

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
            use_decoder: Dict = {"projector": False, "index_decoder_block": -1, "output": False},
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
        # First layer doesn't change shape of input.\
        self.init_filters = init_filters
        self.num_downs_in_encoder = len(blocks_down) - 1
        self.num_ups_in_decoder = len(blocks_up)

        self.use_decoder_projector = use_decoder["projector"]
        self.use_decoder_output = use_decoder["output"]
        self.index = use_decoder["index_decoder_block"]

        num_ups_in_decoder = len(blocks_up)
        if self.use_decoder_projector:
            if -self.index > num_ups_in_decoder or self.index >= 0:
                raise ValueError(f"Decoder has only {num_ups_in_decoder} blocks."
                                 f" [-1, -{num_ups_in_decoder}] indices are valid.")

        if not self.use_decoder_projector and not self.use_decoder_output:
            self.up_layers = None
            self.up_samples = None
            self.conv_final = None

    def freeze_encoder(self):
        encoder_layers = [self.convInit] + self.down_layers
        for param in chain.from_iterable(layer.parameters() for layer in encoder_layers):
            param.requires_grad = False

    def get_encoder_output_size(self, image_size):
        height, width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        height_en_feat = height // 2 ** self.num_downs_in_encoder
        width_en_feat = width // 2 ** self.num_downs_in_encoder
        channels_en_feat = self.init_filters * 2 ** self.num_downs_in_encoder
        return {'channels': channels_en_feat, 'height': height_en_feat, 'width': width_en_feat}

    def get_decoder_output_size(self, image_size, index_decoder_block):
        height, width = image_size if isinstance(image_size, tuple) else (image_size, image_size)

        if -index_decoder_block > self.num_ups_in_decoder or index_decoder_block >= 0:
            raise ValueError(f"Decoder has only {self.num_ups_in_decoder} blocks."
                             f" [-1, -{self.num_ups_in_decoder}] indices are valid.")

        # How many upsample occure from index_decoder_block until end of decoder
        num_ups_until_model_output = abs(1+index_decoder_block)
        height_dec_feat = height // 2 ** num_ups_until_model_output
        width_dec_feat = width // 2 ** num_ups_until_model_output

        # Reverse the encoder filters list to gain decoder filters list.
        up_filters = [self.init_filters * 2 ** up_layer for up_layer in range(self.num_ups_in_decoder)]
        num_ups_until_model_output = abs(1+index_decoder_block)
        channels_dec_feat = up_filters[num_ups_until_model_output]
        return {'channels': channels_dec_feat, 'height': height_dec_feat, 'width': width_dec_feat}

    def forward(self, x, show_size=False):
        x = self.convInit(x)
        if show_size: print(x.size())
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)
            if show_size: print(x.size())

        if self.use_decoder_projector or self.use_decoder_output:
            down_x.reverse()

            for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
                x = up(x) + down_x[i + 1]
                x = upl(x)
                if show_size: print(x.size())
                if self.use_decoder_projector and self.index == [-5, -4, -3, -2, -1][i]:
                    return x

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
                      use_decoder={"projector": True, "index_decoder_block": -1, "output": False},
                      dropout_prob=0.2,
                      ).to(device)

    en_size = model.get_encoder_output_size(image_size=352)
    print('encoder_output_size=', en_size)
    dec_size = model.get_decoder_output_size(image_size=352, index_decoder_block=-4)
    print('decoder_output_size=', dec_size)

    x = torch.ones([3, 1, 352, 352]).to(device)
    out = model(x, show_size=True)
    print('out=', out.size())
