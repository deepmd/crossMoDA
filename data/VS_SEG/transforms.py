import numpy as np
from data.transforms import RandMultiTransformd, RandSampleSlice, Spacing2Dd, AlignCropd, EndOfCache, KeepOriginald
from monai.transforms import \
    Compose, LoadImaged, AddChanneld, NormalizeIntensityd, ScaleIntensityRangePercentilesd, \
    RandFlipd, RandSpatialCropd, RandScaleIntensityd, Orientationd, ToTensord, RandAffined, \
    RandShiftIntensityd, RandGaussianNoised, CenterSpatialCropd


def encoder_train_transforms(opt, domain):
    align_roi = _get_align_roi(domain)
    # Using device=torch.device("cuda") in RandAffine produces some issues:
    # 1. RandFlip, RandScaleIntensity, RandShiftIntensity, RandGaussianNoise all use numpy and having tensor on CUDA
    #    causes error unless make these transforms precede RandAffine in the list.
    # 2. If final tensors are located on CUDA, pin_memory in DataLoader should be False.
    # 3. To use CUDA with multiprocessing, you must use the 'spawn' start method by calling
    #    torch.multiprocessing.set_start_method('spawn'). spawn start method has some requirements which may lead to
    #    some errors such as: "Can't pickle local object" (if any local function are used)
    # 4. Some warnings (Sharing CUDA tensors) and probable speed drop because of multiple processes using CUDA
    return Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacing2Dd(keys=["image"], pixdim=_base_pixdim, mode="bilinear"),
            NormalizeIntensityd(keys=["image"]),
            # N4BiasFieldCorrection
            # ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0, b_max=1),
            AlignCropd(keys=["image"], align_roi=align_roi),
            EndOfCache(),
            RandSampleSlice(keys=["image"]),
            KeepOriginald(keys=["image"], do_transform=opt.debug),
            RandMultiTransformd(keys=["image"], times=2, view_transforms=
                lambda keys: [
                    RandScaleIntensityd(keys=keys, factors=0.1, prob=0.5),  # v = v * (1 + U(-0.1, 0.1))
                    RandShiftIntensityd(keys=keys, offsets=0.1, prob=0.5),  # v = v + U(-0.1, 0.1)
                    RandGaussianNoised(keys=keys, std=0.1, prob=1),  # std=U(0, 0.1) mean=0
                    RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
                    RandAffined(keys=keys, scale_range=0.1, rotate_range=np.pi/18, mode="bilinear", prob=1),
                    RandSpatialCropd(keys=keys, roi_size=(opt.size, opt.size), random_center=True, random_size=False)
                ]
            ),
            ToTensord(keys=["image"])
        ]
    )


def encoder_val_transforms(opt, domain):
    align_roi = _get_align_roi(domain)
    return Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacing2Dd(keys=["image"], pixdim=_base_pixdim, mode="bilinear"),
            NormalizeIntensityd(keys=["image"]),
            AlignCropd(keys=["image"], align_roi=align_roi),
            EndOfCache(),
            KeepOriginald(keys=["image"], do_transform=opt.debug),
            CenterSpatialCropd(keys=["image"], roi_size=(opt.size, opt.size)),
            ToTensord(keys=["image"])
        ]
    )


def supervised_train_transforms(opt, domain):
    align_roi = _get_align_roi(domain)
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacing2Dd(keys=["image", "label"], pixdim=_base_pixdim, mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys=["image"]),
            AlignCropd(keys=["image", "label"], align_roi=align_roi),
            EndOfCache(),
            KeepOriginald(keys=["image"], do_transform=opt.debug),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RandGaussianNoised(keys="image", std=0.1, prob=1),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandAffined(keys=["image", "label"], scale_range=0.1, rotate_range=np.pi/18, mode=("bilinear", "nearest"), prob=1),
            RandSpatialCropd(keys=["image", "label"], roi_size=(opt.size, opt.size), random_center=True, random_size=False),
            ToTensord(keys=["image", "label"])
        ]
    )


def supervised_val_transforms(opt, domain):
    align_roi = _get_align_roi(domain)
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacing2Dd(keys=["image", "label"], pixdim=_base_pixdim, mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys=["image"]),
            AlignCropd(keys=["image", "label"], align_roi=align_roi),
            EndOfCache(),
            KeepOriginald(keys=["image"], do_transform=opt.debug),
            CenterSpatialCropd(keys=["image", "label"], roi_size=(opt.size, opt.size)),
            ToTensord(keys=["image", "label"])
        ]
    )


_base_pixdim = [0.41015625, 0.41015625]


def _get_align_roi(domain):
    if domain == "source":
        return _source_align_roi
    elif domain == "target":
        return _target_align_roi
    else:
        raise ValueError("Value of domain should be either 'source' or 'target'.")


def _source_align_roi(img, meta_dict):
    dim = img.shape[1:4]
    pixdim_z = meta_dict["pixdim"][3]
    cm = 40  # crop margin
    if dim[0] != 512 or dim[1] != 512:
        raise ValueError(f"Unexpected values for dim-xy={dim[:2]}.")
    if dim[2] == 120 and pixdim_z == 1.5:
        return [cm, cm, dim[2]-97], [dim[0]-cm, dim[1]-cm, dim[2]-66]
    elif dim[2] == 160 and pixdim_z == 1:
        return [cm, cm, dim[2]-145], [dim[0]-cm, dim[1]-cm, dim[2]-95]
    else:
        file_name = meta_dict["filename_or_obj"]
        raise ValueError(f"Unexpected values for dim-z={dim[2]} or pixdim-z={pixdim_z} in file {file_name}.")


def _target_align_roi(img, meta_dict):
    dim = img.shape[1:4]
    pixdim_z = meta_dict["pixdim"][3]
    cm = 40  # crop margin
    if dim[0] != 512 or dim[1] != 512:
        raise ValueError(f"Unexpected values for dim-xy={dim[:2]}.")
    if dim[2] == 80 and pixdim_z == 1.5:
        return [cm, cm, dim[2]-70], [dim[0]-cm, dim[1]-cm, dim[2]-35]
    elif dim[2] == 80 and pixdim_z == 1:
        return [cm, cm, dim[2]-65], [dim[0]-cm, dim[1]-cm, dim[2]-10]
    if dim[2] == 40 and pixdim_z == 1.5:
        return [cm, cm, 0], [dim[0]-cm, dim[1]-cm, 40]
    else:
        file_name = meta_dict["filename_or_obj"]
        raise ValueError(f"Unexpected values for dim-z={dim[2]} or pixdim-z={pixdim_z} in file {file_name}.")
