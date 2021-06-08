import torch
from torch import nn
import argparse

from .segresnet import SegResNet
from .res_unet import ResUNet
from .res_unet_plus import ResUNetPlusPlus
from .dr_unet104 import DRUNet104
from .projection import EncoderProjection, DecoderProjection


class SegAuxModel(nn.Module):
    """Creates and return the model

    Args:
        opt: options
        mode: {``"encoder+projector"``,
               ``"encoder+projector+classifier"``,
               ``"encoder+decoder"``,
               ``"encoder+decoder+projector"``}
            Define the mode to create the model.
    """
    def __init__(
            self,
            mode='encoder+decoder',
            base_args={"in_channels": 1,
                       "classes_num": 3,
                       "model": "ResUNet++",
                       "size": 352,
                       },
            enc_proj_args={"head": "mlp", "feat_dim": 128},
            classifiers_args={"outputs_sizes": [8, 2], "detach": [True, True]},
            dec_proj_args={"index_decoder_block": -1,
                           "patch_size": 3,
                           "no_of_patches": 9,
                           "use_final_linear_layer": True,
                           "final_linear_layer_dim": 128,
                           },
    ):
        super(SegAuxModel, self).__init__()
        if mode not in ["encoder+projector", "encoder+projector+classifier",
                        "encoder+decoder", "encoder+decoder+projector"]:
            raise ValueError(f"{mode} is not a valid mode.")
        self.use_encoder_projector = ("encoder+projector" in mode)
        self.use_classifier = ('classifier' in mode)
        self.use_decoder_projector = ('decoder+projector' in mode)
        self.use_decoder_output = ('decoder' in mode and 'projector' not in mode)
        self.detach = classifiers_args["detach"]
        use_decoder = {
            "projector": self.use_decoder_projector,
            "index_decoder_block": dec_proj_args["index_decoder_block"],
            "output": self.use_decoder_output,
        }

        self.base = SegAuxModel.get_base_model(base_args["model"],
                                               base_args,
                                               use_decoder, )

        if self.use_encoder_projector:
            encoder_output_size = self.base.get_encoder_output_size(base_args["size"])
            self.encoder_projector = EncoderProjection(head=enc_proj_args["head"],
                                                       dim_in=encoder_output_size['channels'],
                                                       feat_dim=enc_proj_args["feat_dim"],
                                                       )
        if self.use_classifier:
            classifiers = list([])
            for i, out_size in enumerate(classifiers_args["outputs_sizes"]):
                classifiers.append(
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(
                            in_features=encoder_output_size['channels'],
                            out_features=out_size,
                        ),
                    )
                )
            self.classifiers = nn.Sequential(*classifiers)

        if self.use_decoder_projector:
            decoder_output_size = self.base.get_decoder_output_size(base_args["size"],
                                                                    dec_proj_args["index_decoder_block"])
            self.decoder_projector = DecoderProjection(decoder_feat_shape=decoder_output_size,
                                                       patch_size=dec_proj_args["patch_size"],
                                                       no_of_patches=dec_proj_args["no_of_patches"],
                                                       use_final_linear_layer=dec_proj_args["use_final_linear_layer"],
                                                       final_linear_layer_dim=dec_proj_args["final_linear_layer_dim"],
                                                       )

    @staticmethod
    def get_base_model(base_model_name, model_args, use_decoder):
        if base_model_name == "SegResNet":
            return SegResNet(
                spatial_dims=2,
                init_filters=32,
                in_channels=model_args["in_channels"],
                out_channels=model_args["classes_num"],
                blocks_down=(1, 2, 2, 4, 4, 4),
                blocks_up=(1, 1, 1, 1, 1),
                upsample_mode="deconv",
                use_decoder=use_decoder,
                dropout_prob=0.2,
            )
        elif base_model_name == "ResUNet":
            return ResUNet(
                in_channels=model_args["in_channels"],
                out_channels=model_args["classes_num"],
                filters=(64, 128, 256, 512, 1024),
                use_decoder=use_decoder,
            )
        elif base_model_name == "ResUNet++":
            return ResUNetPlusPlus(
                in_channels=model_args["in_channels"],
                out_channels=model_args["classes_num"],
                filters=(64, 128, 256, 512, 1024, 1024),
                use_decoder=use_decoder,
            )
        elif base_model_name == "DR-UNet104":
            return DRUNet104(
                in_channels=model_args["in_channels"],
                out_channels=model_args["classes_num"],
                init_filters=16,
                layers=[2, 3, 3, 5, 14, 4],
                dropout=0.2,
                use_decoder=use_decoder,
            )
        else:
            raise ValueError(f"Specified model name '{model_args['model']}' is not valid.")

    def forward(self, x):
        results = dict()

        x = self.base(x)
        if self.use_encoder_projector:
            results['encoder_projector'] = self.encoder_projector(x)

        if self.use_classifier:
            for i, (classifier, detach) in enumerate(zip(self.classifiers, self.detach)):
                out = torch.detach(x) if detach else x
                results[f'classifier{i}'] = classifier(out)

        if self.use_decoder_projector:
            results['decoder_projector'] = self.decoder_projector(x)
        elif self.use_decoder_output:
            results['decoder_output'] = x

        return results


if __name__ == '__main__':
    opt = argparse.Namespace()
    opt.in_channels = 1
    opt.classes_num = 2
    opt.n_parts = 8
    opt.size = 352
    opt.model = 'SegResNet'  # ['SegResNet', 'ResUNet', 'ResUNet++', "DR-UNet104"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegAuxModel(mode="encoder+decoder",
                        base_model_args={"in_channels": opt.in_channels,
                                         "classes_num": opt.classes_num,
                                         "model": opt.model,
                                         "size": opt.size,
                                         },
                        classifiers_args={"outputs_sizes": [opt.n_parts, opt.classes_num],
                                          'detach': [True, True]
                                          },
                        dec_proj_args={"index_decoder_block": -1,
                                       "patch_size": 3,
                                       "no_of_patches": 13,
                                       "use_final_linear_layer": True,
                                       "final_linear_layer_dim": 128
                                       },
                        ).to(device)
    x = torch.ones([3, 1, 352, 352]).to(device)
    out = model(x)
    for key, value in out.items():
        print(key, value.size())



