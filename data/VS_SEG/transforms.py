from data.transforms import RandMultiTransformd, RandSampleSlice, Clipd, Spacing2Dd, AlignCropd
from monai.transforms import \
    Compose, LoadImaged, AddChanneld, LabelToMaskd, NormalizeIntensityd, ScaleIntensityRangePercentilesd, \
    RandFlipd, RandSpatialCropd, RandScaleIntensityd, RandAdjustContrastd, Resized, Orientationd, ToTensord


def get_encoder_source_transforms(opt):
    def align_roi(img, meta_dict):
        dim = img.shape[1:4]
        pixdim_z = meta_dict["pixdim"][3]
        if dim[0] != 512 or dim[1] != 512:
            raise ValueError(f"Unexpected values for dim-xy={dim[:2]}.")
        if dim[2] == 120 and pixdim_z == 1.5:
            return [0, 0, 66], [dim[0], dim[1], 97]
        elif dim[2] == 160 and pixdim_z == 1:
            return [0, 0, 95], [dim[0], dim[1], 147]
        else:
            raise ValueError(f"Unexpected values for dim-z={dim[2]} or pixdim-z={pixdim_z}.")

    min_roi_size = [int(s * 0.7) for s in (512, 512)]  # assuming img_size is (512,512), same effect as scale=(0.7, 1)
    return Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacing2Dd(keys=["image"], pixdim=[0.41015625, 0.41015625]),
            NormalizeIntensityd(keys=["image"]),
            # ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0, b_max=1),
            AlignCropd(keys=["image"], align_roi=align_roi),
            RandSampleSlice(keys=["image"]),
            RandMultiTransformd(keys=["image"], times=2, keep_orig=opt.debug, view_transforms=
                lambda keys: [
                    # crop with random size between min_roi_size and img_size
                    RandSpatialCropd(keys=keys, roi_size=min_roi_size, random_center=True, random_size=True),
                    # random flip
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                    # random brightness/intensity and contrast adjustment
                    RandScaleIntensityd(keys=[keys[0]], factors=0.3, prob=1),
                    RandAdjustContrastd(keys=[keys[0]], gamma=(0.7, 1.3), prob=1),
                    Clipd(keys=[keys[0]], minv=0, maxv=1.5),
                    # resize to a fixed size
                    Resized(keys=keys, spatial_size=(opt.size, opt.size))
                ]
            ),
            ToTensord(keys=["image", "part_num"])
        ]
    )


def get_encoder_target_transforms(opt):
    def align_roi(img, meta_dict):
        dim = img.shape[1:4]
        pixdim_z = meta_dict["pixdim"][3]
        if dim[0] != 512 or dim[1] != 512:
            raise ValueError(f"Unexpected values for dim-xy={dim[:2]}.")
        if dim[2] == 80 and pixdim_z == 1.5:
            return [0, 0, 35], [dim[0], dim[1], 70]
        elif dim[2] == 80 and pixdim_z == 1:
            return [0, 0, 10], [dim[0], dim[1], 65]
        if dim[2] == 40 and pixdim_z == 1.5:
            return [0, 0, 0], [dim[0], dim[1], 40]
        else:
            raise ValueError(f"Unexpected values for dim-z={dim[2]} or pixdim-z={pixdim_z}.")

    min_roi_size = [int(s * 0.7) for s in (512, 512)]
    return Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacing2Dd(keys=["image"], pixdim=[0.41015625, 0.41015625]),
            NormalizeIntensityd(keys=["image"]),
            # ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0, b_max=1),
            AlignCropd(keys=["image"], align_roi=align_roi),
            RandSampleSlice(keys=["image"]),
            RandMultiTransformd(keys=["image"], times=2, keep_orig=opt.debug, view_transforms=
                lambda keys: [
                    # crop with random size between min_roi_size and img_size
                    RandSpatialCropd(keys=keys, roi_size=min_roi_size, random_center=True, random_size=True),
                    # random flip
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                    # random brightness/intensity and contrast adjustment
                    RandScaleIntensityd(keys=[keys[0]], factors=0.3, prob=1),
                    RandAdjustContrastd(keys=[keys[0]], gamma=(0.7, 1.3), prob=1),
                    Clipd(keys=[keys[0]], minv=0, maxv=1.5),
                    # resize to a fixed size
                    Resized(keys=keys, spatial_size=(opt.size, opt.size))
                ]
            ),
            ToTensord(keys=["image", "part_num"])
        ]
    )
