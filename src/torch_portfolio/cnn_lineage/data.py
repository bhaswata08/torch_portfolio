import os
from typing import Literal

import torch
import torchvision.datasets as datasets  # pyright: ignore[reportMissingTypeStubs]
import torchvision.transforms.v2 as v2  # pyright: ignore[reportMissingTypeStubs]
from torch.utils.data import DataLoader

from torch_portfolio.cnn_lineage.config import device

type Loader = DataLoader[tuple[torch.Tensor, torch.Tensor]]
type DatasetName = Literal["mnist", "cifar10", "cifar100"]


# ── Transform presets ─────────────────────────────────────────────────────────
def _get_transforms(dataset: DatasetName) -> tuple[v2.Compose, v2.Compose]:
    """Returns (train_transform, test_transform) for the given dataset."""
    base = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]

    match dataset:
        case "mnist":
            train_tf = v2.Compose([*base, v2.AutoAugment()])
            test_tf = v2.Compose(base)

        case "cifar10" | "cifar100":
            normalize = v2.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            )
            upsample = v2.Resize(224)
            train_tf = v2.Compose(
                [
                    *base,
                    upsample,
                    v2.RandomCrop(224, padding=28),
                    v2.RandomHorizontalFlip(),
                    normalize,
                ]
            )
            test_tf = v2.Compose([*base, upsample, normalize])

    return train_tf, test_tf


# ── Dataset registry ──────────────────────────────────────────────────────────
def _load_dataset(dataset: DatasetName, train: bool, transform: v2.Compose):
    root = f"data/{dataset.upper()}"
    match dataset:
        case "mnist":
            return datasets.MNIST(
                root=root, train=train, transform=transform, download=True
            )
        case "cifar10":
            return datasets.CIFAR10(
                root=root, train=train, transform=transform, download=True
            )
        case "cifar100":
            return datasets.CIFAR100(
                root=root, train=train, transform=transform, download=True
            )


# ── Public API ────────────────────────────────────────────────────────────────
def get_loaders(
    dataset: DatasetName,
    batch_size: int = 64,
    num_workers: int | None = None,
) -> tuple[Loader, Loader]:
    """
    Returns (train_loader, test_loader) for the given dataset.
    Handles transforms, pin_memory, and worker count automatically.
    """
    workers = num_workers if num_workers is not None else (os.cpu_count() or 0)
    pin = device.type == "cuda"

    train_tf, test_tf = _get_transforms(dataset)

    train_ds = _load_dataset(dataset, train=True, transform=train_tf)
    test_ds = _load_dataset(dataset, train=False, transform=test_tf)

    train_loader: Loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
    )

    test_loader: Loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
    )

    return train_loader, test_loader


def num_classes(dataset: DatasetName) -> int:
    return {"mnist": 10, "cifar10": 10, "cifar100": 100}[dataset]


def in_channels(dataset: DatasetName) -> int:
    return {"mnist": 1, "cifar10": 3, "cifar100": 3}[dataset]
