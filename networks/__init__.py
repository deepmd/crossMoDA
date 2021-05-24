from .segresnet import SegResNet
from .res_unet import ResUnet
from .res_unet_plus import ResUnetPlusPlus


def get_encoder_model(opt):
    if opt.model == "SegResNet":
        return SegResNet(
            spatial_dims=2,
            init_filters=32,
            in_channels=1,
            out_channels=2,
            blocks_down=(1, 2, 2, 4, 4, 4),
            blocks_up=(1, 1, 1, 1, 1),
            upsample_mode="deconv",
            use_decoder=False,
            dropout_prob=0.2,
            head='mlp',
            feat_dim=128
        )
    elif opt.model == "ResUNet":
        return ResUnet(
            in_channels=1,
            out_channels=2,
            filters=(64, 128, 256, 512, 1024),
            use_decoder=False,
            head='mlp',
            feat_dim=128,
        )
    elif opt.model == "ResUNet++":
        return ResUnetPlusPlus(
            in_channels=1,
            out_channels=2,
            filters=(64, 128, 256, 512, 1024, 1024),
            use_decoder=False,
            head='mlp',
            feat_dim=128,
        )
    else:
        raise ValueError(f"Specified model name '{opt.model}' is not valid.")
