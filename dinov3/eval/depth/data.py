




import logging
import random
from functools import partial
from typing import Any

import numpy as np
import torch

from dinov3.data import make_dataset, make_data_loader, DatasetWithEnumeratedTargets, SamplerType
import dinov3.distributed as distributed


logger = logging.getLogger("dinov3")


def worker_init_fn(worker_id, num_workers, rank, seed):








    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_dataloader(
    transforms: Any,
    dataset_str: str,
    device: int,
    split: str = "train",
    batch_size: int = 1,
    n_gpus: int = 1,
    num_workers: int = 2,
    seed: int = 0,
    use_init_fn=False,
):
















    assert split in ["train", "val", "test"]
    is_train = split == "train"
    ds = make_dataset(dataset_str=dataset_str, transforms=transforms)
    logger.info(f"Dataset {split}:\n{ds}")

    if not is_train:
        assert batch_size == 1, "Evaluation should only be done at batch size 1!"
        ds = DatasetWithEnumeratedTargets(ds, pad_dataset=True, num_replicas=n_gpus)

    if use_init_fn and is_train:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=device, seed=seed + device)
    else:
        init_fn = None
    dataloader = make_data_loader(
        dataset=ds,
        batch_size=batch_size,
        sampler_type=SamplerType.DISTRIBUTED if distributed.is_enabled() else None,
        drop_last=is_train,
        shuffle=is_train,
        persistent_workers=(not is_train),
        worker_init_fn=init_fn,
        seed=seed,
        num_workers=num_workers,
    )

    if is_train:
        return InfiniteDataloader(dataloader)

    return dataloader


class InfiniteDataloader:
    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader
        self.data_iterator = iter(dataloader)
        self.sampler = dataloader.sampler
        if not hasattr(self.sampler, "epoch"):
            self.sampler.epoch = 0  

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self.dataloader)

    def __next__(self):
        try:
            data = next(self.data_iterator)
        except StopIteration:
            self.sampler.epoch += 1
            self.data_iterator = iter(self.dataloader)
            data = next(self.data_iterator)
        return data
