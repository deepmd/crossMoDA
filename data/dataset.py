from copy import deepcopy

import torch
from monai.data.dataset import CacheDataset
from monai.utils import ensure_tuple
from monai.config.type_definitions import KeysCollection
from typing import Callable, List, Optional, Sequence, Union, Tuple, Dict
import numpy as np
from math import ceil


class CachePartDataset(CacheDataset):
    def __init__(
            self,
            data: Sequence,
            transform: Union[Sequence[Callable], Callable],
            parts_num: int,
            keys: KeysCollection,
            num_workers: Optional[int] = None,
            progress: bool = True,
            copy_other_keys: bool = True
    ) -> None:
        super().__init__(data=data, transform=transform, num_workers=num_workers, progress=progress)
        self.keys = ensure_tuple(keys)
        self.copy_other_keys = copy_other_keys
        self.parts_num = parts_num
        self._cache = self._split_cache(self._cache, parts_num)
        self.cache_num = len(self._cache)
        self.data = [x for x in self.data for _ in range(parts_num)]

    def _split_cache(self, cache: List, parts_num: int) -> List:
        split_cache = []
        for item in cache:
            split_items = dict()
            for key in self.keys:
                if isinstance(item[key], torch.Tensor):
                    split_items[key] = torch.split(item[key], ceil(item[key].shape[-1]/parts_num), dim=-1)
                elif isinstance(item[key], np.ndarray):
                    split_items[key] = np.array_split(item[key], parts_num, axis=-1)
                else:
                    raise TypeError(f"Cannot split data of type {type(item[key]).__name__}.")
            split_items['part_idx'] = np.arange(parts_num)  # list(range(parts_num)) has issues, if used in ToTensord
            if self.copy_other_keys:
                for other_key in item.keys():
                    if other_key not in self.keys:
                        split_items[other_key] = [deepcopy(item[other_key]) for _ in range(parts_num)]
            # "dict of lists" to "list of dicts" {0:[4,5,6], 1:[7,8,9]} --> [{0:4, 1:7}, {0:5, 1:8}, {0:6, 1:9}]
            split_items = [{k: v[i] for k, v in split_items.items()} for i in range(parts_num)]
            split_cache.extend(split_items)
        return split_cache


class CacheSliceDataset(CacheDataset):
    def __init__(
            self,
            data: Sequence,
            transform: Union[Sequence[Callable], Callable],
            keys: KeysCollection,
            stats_collector: Optional[Callable[[Dict], Dict]] = None,
            num_workers: Optional[int] = None,
            progress: bool = True,
            copy_other_keys: bool = True
    ) -> None:
        super().__init__(data=data, transform=transform, num_workers=num_workers, progress=progress)
        self.keys = ensure_tuple(keys)
        self.copy_other_keys = copy_other_keys
        self.stats_collector = stats_collector
        self._cache, slices_nums = self._slice_cache(self._cache)
        self.cache_num = len(self._cache)
        self.data = [x for i, x in enumerate(self.data) for _ in range(slices_nums[i])]

    def _slice_cache(self, cache: List) -> Tuple[List, List]:
        slice_cache = []
        slices_nums = []
        for idx, item in enumerate(cache):
            # collect stats
            stat_keys = []
            if self.stats_collector is not None:
                stats_dict = self.stats_collector(item)
                for stat_key, stat_value in stats_dict.items():
                    if stat_key in item:
                        raise KeyError(f"Key {stat_key} already exists in data.")
                    item[stat_key] = stat_value
                    stat_keys.append(stat_key)
            stat_keys = ensure_tuple(stat_keys)
            # divide to slices
            slices = dict()
            slices_num = item[self.keys[0]].shape[-1]
            for key in (self.keys + stat_keys):
                if item[key].shape[-1] != slices_num:
                    raise ValueError(f"Different number of slices exist in key '{key}'.")
                if isinstance(item[key], torch.Tensor) or isinstance(item[key], np.ndarray):
                    slices[key] = [item[key][..., i] for i in range(slices_num)]
                else:
                    raise TypeError(f"Cannot divide slices of type {type(item[key]).__name__}.")
            slices['slice_idx'] = np.arange(slices_num)  # list(range(slices_num)) has issues, if used in ToTensord
            slices['vol_idx'] = np.repeat(idx, slices_num)  # [idx]*slices_num has issues, if used in ToTensord
            if self.copy_other_keys:
                for other_key in item.keys():
                    if other_key not in (self.keys + stat_keys):
                        slices[other_key] = [deepcopy(item[other_key]) for _ in range(slices_num)]
            # "dict of lists" to "list of dicts" {0:[4,5,6], 1:[7,8,9]} --> [{0:4, 1:7}, {0:5, 1:8}, {0:6, 1:9}]
            slices = [{k: v[i] for k, v in slices.items()} for i in range(slices_num)]
            slice_cache.extend(slices)
            slices_nums.append(slices_num)
        return slice_cache, slices_nums
