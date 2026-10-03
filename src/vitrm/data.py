"""CIFAR-10 / CIFAR-100 data loaders."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vitrm.config import DatasetConfig

# CIFAR-10 statistics, used for both CIFAR-10 and CIFAR-100.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

DATASETS = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}


def train_transform(randaugment: bool) -> transforms.Compose:
    ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if randaugment:
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


def eval_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def build_loaders(
    cfg: DatasetConfig,
    batch_size: int,
    num_workers: int = 4,
    randaugment: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, test_loader)``; the dataset is downloaded to ``cfg.root`` if needed."""
    dataset_cls = DATASETS[cfg.name]
    train_set = dataset_cls(
        cfg.root, train=True, download=True, transform=train_transform(randaugment)
    )
    test_set = dataset_cls(cfg.root, train=False, download=True, transform=eval_transform())
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    return train_loader, test_loader
