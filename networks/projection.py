import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EncoderProjection(nn.Module):
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


class DecoderProjection(nn.Module):
    def __init__(
            self,
            decoder_feat_shape,
            patch_size=3,
            no_of_patches=9,
            use_final_linear_layer=True,
            final_linear_layer_dim=128,
    ):
        super(DecoderProjection, self).__init__()
        assert patch_size in [1, 3], "Only valid values for patch sizes are 1, 3."
        assert no_of_patches in [9, 13], "Only valid values for number of patches are 9, 13."
        channels = decoder_feat_shape["channels"]
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=False)
        self.patch_cropper = CropPatches(decoder_feat_shape, patch_size, no_of_patches)
        self.use_final_linear_layer = use_final_linear_layer
        if self.use_final_linear_layer:
            no_in_filters = channels * patch_size * patch_size
            self.final_linear_layer = nn.Linear(no_in_filters, final_linear_layer_dim, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.patch_cropper(x)
        if self.use_final_linear_layer:
            x = self.final_linear_layer(x)

        return x


class CropPatches(nn.Module):
    def __init__(
            self,
            decoder_feat_shape,
            patch_size=3,
            no_of_patches=9,
    ):
        super(CropPatches, self).__init__()
        self.patch_size = patch_size
        self.no_of_patches = no_of_patches
        self.height_dec_feat = decoder_feat_shape["height"]
        self.width_dec_feat = decoder_feat_shape["width"]

        # Set height and width margin base on patch_size to crop patches safely without out of bound error.
        margin = 4 if self.patch_size == 3 else 1
        marginal_height_dec_feat = self.height_dec_feat - margin
        marginal_width_dec_feat = self.width_dec_feat - margin

        # Get positional indices for the patches to be selected for computing local contrastive loss.
        indices = self.get_indices(marginal_height_dec_feat,
                                   marginal_width_dec_feat,
                                   self.no_of_patches,
                                   self.patch_size)
        self.columns_indices, self.rows_indices = indices

    @staticmethod
    def get_indices(height_feat, width_feat, no_of_patches, patch_size):
        height_indices = [np.linspace(0, height_feat, 3, dtype=np.int)]
        width_indices = [np.linspace(0, width_feat, 3, dtype=np.int)]
        if no_of_patches == 13:
            height_indices.append([height_feat // 4, 3 * height_feat // 4])
            width_indices.append([width_feat // 4, 3 * width_feat // 4])

        columns_indices = [
                [int(index+i) for index in sublist for i in range(patch_size)]
                for sublist in height_indices
            ]
        rows_indices = [
            [int(index+i) for index in sublist for i in range(patch_size)]
            for sublist in width_indices
        ]
        return columns_indices, rows_indices

    def crop_patches(self, in_features):
        batch_size, channels, x_feat_size, y_feat_size = in_features.size()
        assert x_feat_size == self.height_dec_feat
        assert y_feat_size == self.width_dec_feat
        assert len(self.columns_indices) == len(self.rows_indices)

        patches_list = list([])
        for columns_indices, rows_indices in zip(self.columns_indices, self.rows_indices):
            columns_indices = torch.tensor(columns_indices, device=in_features.device)
            rows_indices = torch.tensor(rows_indices, device=in_features.device).unsqueeze(0).t()

            columns_indices = columns_indices.repeat(batch_size, channels, x_feat_size, 1)
            patches = torch.gather(in_features, dim=3, index=columns_indices)
            rows_indices = rows_indices.repeat(batch_size, channels, 1, patches.size(-1))
            patches = torch.gather(patches, dim=2, index=rows_indices)

            patches = F.unfold(patches, kernel_size=self.patch_size, stride=self.patch_size)
            patches = patches.view([batch_size, channels, self.patch_size, self.patch_size, patches.size(-1)])
            patches = patches.permute([0, 4, 1, 2, 3])
            patches = torch.flatten(patches, start_dim=2)

            patches_list.append(patches)

        patches = torch.cat(patches_list, dim=1)
        return patches

    def forward(self, x):
        return self.crop_patches(x)


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    cropper = CropPatches(decoder_feat_shape={"channels": 512, "height": 44, "width": 44},
                          patch_size=1,
                          no_of_patches=9,
                          ).to(device)
    # projection = DecoderProjection(decoder_feat_shape={'channels': 512, 'height': 44, 'width': 44},
    #                                patch_size=3,
    #                                no_of_patches=9,
    #                                use_final_linear_layer=False,
    #                                final_linear_layer_dim=128,
    #                                ).to(device)
    x = torch.ones([16, 512, 44, 44]).to(device)
    cropped_patches = cropper(x)
    print('patches=', cropped_patches.size())