from typing import Any, cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pad: nn.ZeroPad2d = nn.ZeroPad2d(2)
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=6,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
        )  # Output: floor((W_1 - F + 2P)/S) + 1 = floor((32 - 5)/1) + 1 = 28
        self.pool1: nn.AvgPool2d = nn.AvgPool2d(
            kernel_size=(2, 2), stride=(2, 2)
        )  # output: floor((28 - 2)/2)+1 = 14
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
        )  # output: floor((14-5)/1)+1 = 10
        # Reuse pool 1, output: floor((10-2)/2) + 1 = 5
        self.conv3: nn.Conv2d = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(5, 5),
            stride=(1, 1),
        )  # output: floor((5-5)/1) + 1 = 1
        self.flatten = nn.Flatten(1)
        self.fc1: nn.Linear = nn.Linear(120, 84)
        self.fc2: nn.Linear = nn.Linear(84, 10)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv1(x)
        x = F.tanh(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.tanh(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.tanh(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.tanh(x)
        return cast(torch.Tensor, self.fc2(x))
