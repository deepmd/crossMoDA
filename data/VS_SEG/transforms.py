import numpy as np
import torch
from data.transforms import RandMultiTransformd, RandSampleSlice, Clipd, Spacing2Dd, AlignCropd
from monai.transforms import \
    Compose, LoadImaged, AddChanneld, NormalizeIntensityd, ScaleIntensityRangePercentilesd, \
    RandFlipd, RandSpatialCropd, RandScaleIntensityd, Orientationd, ToTensord, RandAffined, \
    RandShiftIntensityd, RandGaussianNoised


def get_encoder_source_transforms(opt):
    def align_roi(img, meta_dict):
        dim = img.shape[1:4]
        pixdim_z = meta_dict["pixdim"][3]
        cm = 40  # crop margin
        if dim[0] != 512 or dim[1] != 512:
            raise ValueError(f"Unexpected values for dim-xy={dim[:2]}.")
        if dim[2] == 120 and pixdim_z == 1.5:
            return [cm, cm, 66], [dim[0]-cm, dim[1]-cm, 97]
        elif dim[2] == 160 and pixdim_z == 1:
            return [cm, cm, 95], [dim[0]-cm, dim[1]-cm, 147]
        else:
            raise ValueError(f"Unexpected values for dim-z={dim[2]} or pixdim-z={pixdim_z}.")

    # Using device=torch.device("cuda") in RandAffine produces two issues:
    # 1. RandFlip, RandScaleIntensity, RandShiftIntensity, RandGaussianNoise all use numpy and having tensor on CUDA
    #    causes error unless make these transforms precede RandAffine in the list.
    # 2. If final tensors are located on CUDA, pin_memory in DataLoader should be False.
    return Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacing2Dd(keys=["image"], pixdim=[0.41015625, 0.41015625]),
            NormalizeIntensityd(keys=["image"]),
            # N4BiasFieldCorrection
            # ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0, b_max=1),
            AlignCropd(keys=["image"], align_roi=align_roi),
            RandSampleSlice(keys=["image"]),
            RandMultiTransformd(keys=["image"], times=2, keep_orig=opt.debug, view_transforms=
                lambda keys: [
                    RandScaleIntensityd(keys=keys[0], factors=0.1, prob=0.5),  # v = v * (1 + U(-0.1, 0.1))
                    RandShiftIntensityd(keys=keys[0], offsets=0.1, prob=0.5),  # v = v + U(-0.1, 0.1)
                    RandGaussianNoised(keys=keys[0], std=0.1, prob=1),  # std=U(0, 0.1) mean=0
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                    RandAffined(keys=keys, scale_range=0.1, rotate_range=np.pi/18, prob=1, device=torch.device("cuda")),
                    RandSpatialCropd(keys=keys, roi_size=(opt.size, opt.size), random_center=True, random_size=False)
                ]
            ),
            ToTensord(keys=["image", "part_num"])
        ]
    )


def get_encoder_target_transforms(opt):
    def align_roi(img, meta_dict):
        dim = img.shape[1:4]
        pixdim_z = meta_dict["pixdim"][3]
        cm = 40  # crop margin
        if dim[0] != 512 or dim[1] != 512:
            raise ValueError(f"Unexpected values for dim-xy={dim[:2]}.")
        if dim[2] == 80 and pixdim_z == 1.5:
            return [cm, cm, 35], [dim[0]-cm, dim[1]-cm, 70]
        elif dim[2] == 80 and pixdim_z == 1:
            return [cm, cm, 10], [dim[0]-cm, dim[1]-cm, 65]
        if dim[2] == 40 and pixdim_z == 1.5:
            return [cm, cm, 0], [dim[0]-cm, dim[1]-cm, 40]
        else:
            raise ValueError(f"Unexpected values for dim-z={dim[2]} or pixdim-z={pixdim_z}.")

    return Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacing2Dd(keys=["image"], pixdim=[0.41015625, 0.41015625]),
            NormalizeIntensityd(keys=["image"]),
            # N4BiasFieldCorrection
            # ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0, b_max=1),
            AlignCropd(keys=["image"], align_roi=align_roi),
            RandSampleSlice(keys=["image"]),
            RandMultiTransformd(keys=["image"], times=2, keep_orig=opt.debug, view_transforms=
                lambda keys: [
                    RandScaleIntensityd(keys=keys[0], factors=0.1, prob=0.5),  # v = v * (1 + U(-0.1, 0.1))
                    RandShiftIntensityd(keys=keys[0], offsets=0.1, prob=0.5),  # v = v + U(-0.1, 0.1)
                    RandGaussianNoised(keys=keys[0], std=0.1, prob=1),  # std=U(0, 0.1) mean=0
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                    RandAffined(keys=keys, scale_range=0.1, rotate_range=np.pi/18, prob=1, device=torch.device("cuda")),
                    RandSpatialCropd(keys=keys, roi_size=(opt.size, opt.size), random_center=True, random_size=False)
                ]
            ),
            ToTensord(keys=["image", "part_num"])
        ]
    )
