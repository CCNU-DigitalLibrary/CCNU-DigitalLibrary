# encoding: utf-8
import itertools
from typing import Optional

import numpy as np
from torch.utils.data import Sampler

from lib.utils import comm


class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._can_reuse_indices = False

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        self.epoch = 0

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        start = self._rank
        if self._can_reuse_indices:
            _indices = self._indices
            self._can_reuse_indices = False
        else:
            _indices = self._prepare_indices()
        self._indices = _indices

        yield from itertools.islice(self._indices, start, None, self._world_size)

    def _prepare_indices(self):
        np.random.seed(self._seed + self.epoch)
        if self._shuffle:
            return np.random.permutation(self._size)
        else:
            return np.arange(self._size)

    def __len__(self):
        if not hasattr(self, '_indices'):
            self._indices = self._prepare_indices()
            self._can_reuse_indices = True
        return len(self._indices) // self._world_size

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

class InferenceSampler(Sampler):
    """
    Produce indices for inference.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        shard_size = (self._size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
