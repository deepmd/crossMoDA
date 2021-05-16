from torch.utils.data import Sampler, RandomSampler, SequentialSampler
from typing import List, Iterable
from itertools import zip_longest


class MultiDomainBatchSampler(Sampler[List[int]]):
    def __init__(self,
                 domains_lengths: Iterable[int],
                 batch_size: int,
                 parts_num: int,
                 batch_parts_num: int = 0,
                 drop_last: bool = False,
                 shuffle: bool = True,
                 replacement: bool = False) -> None:
        self.domains_num = len(tuple(domains_lengths))
        self.batch_size = batch_size
        self.parts_num = parts_num
        self.drop_last = drop_last
        self.domain_batch_size = self.batch_size // self.domains_num
        self.batch_parts_num = batch_parts_num if batch_parts_num > 0 else self.parts_num
        self.batch_vols_num = self.domain_batch_size // self.batch_parts_num
        self.domains_parts_num = [length*parts_num for length in domains_lengths]
        if self.batch_size % self.domains_num != 0:
            raise ValueError(f"Batch size ({self.batch_size}) should be divisible by "
                             f"number of domains ({self.domains_num}).")
        if self.domain_batch_size % self.batch_parts_num != 0:
            raise ValueError(f"Domain batch size ({self.domain_batch_size}) should be divisible by "
                             f"batch number of partitions (batch_parts_num={self.batch_parts_num}).")
        if self.parts_num % self.batch_parts_num != 0:
            raise ValueError(f"Total number of partitions (parts_num={parts_num}) should be divisible by "
                             f"batch number of partitions (batch_parts_num={self.batch_parts_num}).")
        max_domain_length = max(*domains_lengths)
        self.vol_sampler = RandomSampler([None] * max_domain_length, replacement) if shuffle else \
                           SequentialSampler([None] * max_domain_length)
        self.part_sampler = RandomSampler([None] * parts_num, replacement) if shuffle else \
                            SequentialSampler([None] * parts_num)

    def __iter__(self):
        # Iterates batch_vols_num items at a time from vol_sampler
        for vol_idxs in zip_longest(*[iter(self.vol_sampler)]*self.batch_vols_num, fillvalue=None):
            if None not in vol_idxs or not self.drop_last:
                domain_batch = []
                # Iterates batch_parts_num items at a time from part_sampler
                for part_idxs in zip(*[iter(self.part_sampler)]*self.batch_parts_num):
                    domain_batch = [(vi * self.parts_num + pi) for vi in vol_idxs for pi in part_idxs if vi is not None]
                batch = []
                cumsum_parts_num = 0
                for domain_parts_num in self.domains_parts_num:
                    batch.extend([(idx % domain_parts_num + cumsum_parts_num) for idx in domain_batch])
                    cumsum_parts_num += domain_parts_num
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.vol_sampler) // self.batch_vols_num
        else:
            return (len(self.vol_sampler) + self.batch_vols_num - 1) // self.batch_vols_num
