from itertools import chain
from typing import Mapping, Hashable, Dict, Sequence, Optional, Callable, Union
import numpy as np
import torch
from copy import deepcopy
from monai.config import KeysCollection, DtypeLike
from monai.transforms.spatial.dictionary import GridSampleModeSequence, GridSamplePadModeSequence
from monai.utils import GridSampleMode, GridSamplePadMode, ensure_tuple
from monai.transforms import \
    MapTransform, Randomizable, Compose, ConcatItemsd, DeleteItemsd, Lambdad, Spacingd, Lambda


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
        d: Dict = dict(data)
        meta_data_key = f"{self.keys[0]}_{self.meta_key_postfix}"
        meta_data = d[meta_data_key]
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
        d: Dict = dict(data)
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
            d[key] = np.clip(d[key], a_min=self.minv, a_max=self.maxv)
        return d


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
            keep_orig: bool = False,
            allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        if times < 1:
            raise ValueError(f"times must be positive, got {times}.")
        self.times = times
        self.views_keys = {k: [f"{k}_{i+1}" for i in range(times)] for k in keys}
        if view_transforms is None:
            view_transforms = []
        else:
            keys_in_views = [[self.views_keys[k][i] for k in keys] for i in range(times)]
            view_transforms = [Compose(view_transforms(keys)) for keys in keys_in_views]
        all_views_keys = list(chain.from_iterable(self.views_keys.values()))
        # shallow copy of original keys (image -> image_orig, ...) for debugging purposes
        if keep_orig:
            def copy_orig(d):
                for key in self.views_keys.keys():
                    d.update({key+"_orig": d[key]})
                return d
            view_transforms.append(Lambda(func=copy_orig))
        # Adds a new dimension at 'dim' to all views (image_1, ...)
        view_transforms.append(Lambdad(keys=all_views_keys, func=lambda d: np.expand_dims(d, axis=view_dim)))
        # Concatenates views on dimension 'dim' and writes the result over original keys cat(image_1, image_2) -> image
        view_transforms.extend(
            [ConcatItemsd(keys=keys, name=orig_key, dim=view_dim) for (orig_key, keys) in self.views_keys.items()])
        # Deletes all views (image_1, ...)
        view_transforms.append(DeleteItemsd(keys=all_views_keys))
        Compose.__init__(self, view_transforms)

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
