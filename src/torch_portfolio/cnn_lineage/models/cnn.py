from typing import Any, cast, override

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1: nn.Linear = nn.Linear(16 * 7 * 7, num_classes)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cast(torch.Tensor, self.conv1(x))
        x = cast(torch.Tensor, self.pool(x))
        x = cast(torch.Tensor, self.conv2(x))
        x = cast(torch.Tensor, self.pool(x))
        x = cast(torch.Tensor, self.fc1(x))
        return x
