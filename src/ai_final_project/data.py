from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def build_transforms(data_config: dict[str, Any]) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=data_config["normalize_mean"],
        std=data_config["normalize_std"],
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def create_cifar10_dataloaders(
    data_config: dict[str, Any],
    num_workers: int,
    seed: int,
) -> dict[str, DataLoader]:
    train_transform, eval_transform = build_transforms(data_config)
    root = data_config["root"]

    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    eval_train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=eval_transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=eval_transform)

    val_size = int(len(train_dataset) * data_config["val_split"])
    train_size = len(train_dataset) - val_size
    train_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed)

    train_subset, _ = random_split(train_dataset, [train_size, val_size], generator=train_generator)
    _, val_subset = random_split(eval_train_dataset, [train_size, val_size], generator=val_generator)

    batch_size = data_config["batch_size"]
    pin_memory = torch.cuda.is_available()

    return {
        "train": DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
