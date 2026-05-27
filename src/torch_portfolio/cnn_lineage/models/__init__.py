from typing import Callable

import torch.nn as nn

from torch_portfolio.cnn_lineage.models.cnn import CNN
from torch_portfolio.cnn_lineage.models.fcn import FCN

type ModelFactory = Callable[[int, int], nn.Module]

_REGISTRY: dict[str, ModelFactory] = {
    "fcn": lambda c, n: FCN(input_size=c * 28 * 28, num_classes=n),
    "cnn": lambda c, n: CNN(in_channels=c, num_classes=n),
}


def get_model(name: str, in_channels: int, num_classes: int) -> nn.Module:
    name = name.lower()
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY)
        raise ValueError(f"Unknown model {name!r}, Available: {available}")
    return _REGISTRY[name](in_channels, num_classes)


def list_models() -> list[str]:
    return list(_REGISTRY)
