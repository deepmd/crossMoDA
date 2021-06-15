from itertools import chain
from typing import Mapping, Hashable, Dict, Sequence, Optional, Callable, Union, Any
import numpy as np
import torch
from copy import deepcopy
from monai.config import KeysCollection, DtypeLike
from monai.transforms.spatial.dictionary import GridSampleModeSequence, GridSamplePadModeSequence
from monai.utils import GridSampleMode, GridSamplePadMode, ensure_tuple
from monai.transforms import \
    MapTransform, Randomizable, Compose, ConcatItemsd, DeleteItemsd, Lambdad, Spacingd, Transform, CopyItemsd


class Spacing2Dd(Spacingd):
    def __init__(
            self,
            keys: KeysCollection,
            pixdim: Sequence[float],
            diagonal: bool = False,
            mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
            padding_mode: GridSamplePadModeSequence = GridSamplePadMode.BORDER,
            align_corners: Union[Sequence[bool], bool] = False,
            dtype: Optional[Union[Sequence[DtypeLike], DtypeLike]] = np.float64,
            meta_key_postfix: str = "meta_dict",
            allow_missing_keys: bool = False,
    ) -> None:
        if len(pixdim) != 2:
            raise ValueError(f"pixdim must have two values, got {len(pixdim)}.")
        super(Spacing2Dd, self).__init__(keys, pixdim, diagonal, mode, padding_mode, align_corners, dtype,
                                         meta_key_postfix, allow_missing_keys)

    def __call__(
            self, data: Mapping[Union[Hashable, str], Dict[str, np.ndarray]]
    ) -> Dict[Union[Hashable, str], Union[np.ndarray, Dict[str, np.ndarray]]]:
        meta_data_key = f"{self.keys[0]}_{self.meta_key_postfix}"
        meta_data = data[meta_data_key]
        pixdim_z = meta_data["pixdim"][3]
        pixdim = self.spacing_transform.pixdim
        self.spacing_transform.pixdim = np.array([pixdim[0], pixdim[1], pixdim_z])
        return super(Spacing2Dd, self).__call__(data)


class AlignCropd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        align_roi: Callable[[np.ndarray, Dict], Sequence[Union[Sequence[int], np.ndarray]]],
        meta_key_postfix: str = "meta_dict",
        allow_missing_keys: bool = False,
    ) -> None:
        super(AlignCropd, self).__init__(keys, allow_missing_keys)
        self.align_roi = align_roi
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            meta_data_key = f"{key}_{self.meta_key_postfix}"
            meta_data = d[meta_data_key]
            align_roi = self.align_roi(d[key], meta_data)
            align_roi = ensure_tuple(align_roi)
            if len(align_roi) != 2:
                raise ValueError(f"The return value of align_roi must have two values, got {len(align_roi)}.")
            roi_start_np = np.maximum(np.asarray(align_roi[0], dtype=np.int16), 0)
            roi_end_np = np.maximum(np.asarray(align_roi[1], dtype=np.int16), roi_start_np)
            # assuming `img` is channel-first and slicing doesn't apply to the channel dim.
            slices = [slice(None)] + [slice(s, e) for s, e in zip(roi_start_np, roi_end_np)]
            d[key] = d[key][tuple(slices)]
        return d


class Clipd(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            minv: float,
            maxv: float,
            allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.minv = minv
        self.maxv = maxv

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            if isinstance(d[key], torch.Tensor):
                d[key] = torch.clip(d[key], min=self.minv, max=self.maxv)
            elif isinstance(d[key], np.ndarray):
                d[key] = np.clip(d[key], a_min=self.minv, a_max=self.maxv)
            else:
                raise TypeError(f"Cannot clip data of type {type(d[key]).__name__}.")
        return d


class EndOfCache(Randomizable, Transform):
    """
    Return unchanged input data. It can be used to separate cached transforms from non-cached transforms.
    """

    def randomize(self, data: Any) -> None:
        pass

    def __call__(self, data: Any) -> Any:
        return data


class KeepOriginald(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            original_key_postfix: str = "orig",
            do_transform: bool = True,
            allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.original_key_postfix = original_key_postfix
        self.do_transform = do_transform
        if self.do_transform:
            orig_names = [f"{key}_{self.original_key_postfix}" for key in self.keys]
            self.copier = CopyItemsd(keys=keys, times=1, names=orig_names)

    def __call__(self, data: Any) -> Any:
        if self.do_transform:
            return self.copier.__call__(data)
        return data


class RandSampleSlice(Randomizable, MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            allow_missing_keys: bool = False) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self._pos = None

    def randomize(self, slices_num: int) -> None:
        self._pos = self.R.randint(0, slices_num)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize(d[self.keys[0]].shape[-1])
        if self._pos is None:
            raise AssertionError
        for key in self.key_iterator(d):
            d[key] = d[key][..., self._pos]
        return d


class RandMultiTransformd(Compose, MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            times: int,
            view_transforms: Optional[Callable[[KeysCollection], Sequence[Callable]]] = None,
            view_dim: int = 0,
            allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        if times < 1:
            raise ValueError(f"times must be positive, got {times}.")
        self.times = times
        self.view_dim = view_dim
        self.views_keys = {k: [f"{k}_{i+1}" for i in range(times)] for k in keys}
        if view_transforms is None:
            view_transforms = []
        else:
            keys_in_views = [[self.views_keys[k][i] for k in keys] for i in range(times)]
            view_transforms = [Compose(view_transforms(keys)) for keys in keys_in_views]
        all_views_keys = list(chain.from_iterable(self.views_keys.values()))
        # Adds a new dimension at 'view_dim' to all views (image_1, ...)
        view_transforms.append(Lambdad(keys=all_views_keys, func=self._add_dim))
        # Concatenates views on dimension 'view_dim' and writes the result over original keys cat(image_1, image_2) -> image
        view_transforms.extend(
            [ConcatItemsd(keys=keys, name=orig_key, dim=self.view_dim) for (orig_key, keys) in self.views_keys.items()])
        # Deletes all views (image_1, ...)
        view_transforms.append(DeleteItemsd(keys=all_views_keys))

        Compose.__init__(self, view_transforms)

    def _add_dim(self, d):
        if isinstance(d, torch.Tensor):
            return torch.unsqueeze(d, dim=self.view_dim)
        elif isinstance(d, np.ndarray):
            return np.expand_dims(d, axis=self.view_dim)
        else:
            raise TypeError(f"Cannot expand dimension of type {type(d).__name__}.")

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            for new_key in self.views_keys[key]:
                if new_key in d:
                    raise KeyError(f"Key {new_key} already exists in data.")
                if isinstance(d[key], torch.Tensor):
                    d[new_key] = d[key].detach().clone()
                else:
                    d[new_key] = deepcopy(d[key])
        d = Compose.__call__(self, d)
        return d


class MapSegLabelToClassLabel(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            num_classes: int,
            include_background: bool = False,
            allow_missing_keys: bool = False) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.num_classes = num_classes
        self.include_background = include_background

    def _map(self, img):
        labels = range(self.num_classes) if self.include_background else range(1, self.num_classes)
        if isinstance(img, torch.Tensor):
            return torch.tensor([torch.any(img == label) for label in labels], dtype=img.dtype)
        elif isinstance(img, np.ndarray):
            return np.array([np.any(img == label) for label in labels], dtype=img.dtype)
        else:
            raise TypeError(f"Cannot map segmentation labels to classification labels of type {type(img).__name__}.")

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self._map(d[key])
        return d


class AddPartitionIndex(Transform):
    def __init__(
            self,
            part_idx_key: str,
            parts_num: int,
            vol_size_key: str = "vol_size",
            slice_idx_key: str = "slice_idx") -> None:
        self.parts_num = parts_num
        self.part_idx_key = part_idx_key
        self.vol_size_key = vol_size_key
        self.slice_idx_key = slice_idx_key

    def _compute_part_idx(self, vol_size, slice_idx):
        Neach_section, extras = divmod(vol_size, self.parts_num)
        section_sizes = np.array(extras * [Neach_section + 1] + (self.parts_num - extras) * [Neach_section])
        div_points = np.cumsum(section_sizes)
        part_idx = (slice_idx >= div_points).sum()
        return part_idx

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        d[self.part_idx_key] = self._compute_part_idx(d[self.vol_size_key], d[self.slice_idx_key])
        return d
