from .segresnet import SegResNet
from .res_unet import ResUNet
from .res_unet_plus import ResUNetPlusPlus
from .dr_unet104 import DRUNet104


def get_model(opt, mode="encoder+decoder"):
    """Creates and return the model

    Args:
        opt: options
        mode: {``"encoder+decoder"``, ``"encoder+projection"``, ``"encoder+decoder+projection"``}
            Define the mode to create the model.
    """
    if opt.model == "SegResNet":
        return SegResNet(
            spatial_dims=2,
            init_filters=32,
            in_channels=opt.in_channels,
            out_channels=opt.classes_num,
            blocks_down=(1, 2, 2, 4, 4, 4),
            blocks_up=(1, 1, 1, 1, 1),
            upsample_mode="deconv",
            use_decoder=("decoder" in mode),
            dropout_prob=0.2,
            head='mlp',
            feat_dim=128
        )
    elif opt.model == "ResUNet":
        return ResUNet(
            in_channels=opt.in_channels,
            out_channels=opt.classes_num,
            filters=(64, 128, 256, 512, 1024),
            use_decoder=("decoder" in mode),
            head='mlp',
            feat_dim=128
        )
    elif opt.model == "ResUNet++":
        return ResUNetPlusPlus(
            in_channels=opt.in_channels,
            out_channels=opt.classes_num,
            filters=(64, 128, 256, 512, 1024, 1024),
            use_decoder=("decoder" in mode),
            head='mlp',
            feat_dim=128
        )
    elif opt.model == "DR-UNet104":
        return DRUNet104(
            in_channels=opt.in_channels,
            out_channels=opt.classes_num,
            init_filters=16,
            layers=[2, 3, 3, 5, 14, 4],
            dropout=0.2,
            use_decoder=("decoder" in mode),
            head='mlp',
            feat_dim=128
        )
    else:
        raise ValueError(f"Specified model name '{opt.model}' is not valid.")
