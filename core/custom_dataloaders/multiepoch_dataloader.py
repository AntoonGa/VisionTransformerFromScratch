"""
Author: yoniaflalo }
Features:
This brilliant dataloader is from https://github.com/huggingface/pytorch-image-models/pull/140
This keeps the workers alive between epochs and nullifies the waiting time between epochs
"""
import torch
from torch.utils.data import DataLoader


class MultiEpochsDataLoader(DataLoader):
    """ pass a dataset to this dataloader and use it as a normal dataloader"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> tuple[torch.Tensor, int]:
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler) -> None:
        self.sampler = sampler

    def __iter__(self):  # noqa: ANN204
        while True:
            yield from iter(self.sampler)
