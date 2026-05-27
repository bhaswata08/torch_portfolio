from typing import Any, override

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int, *args: Any, **kwargs: Any
    ) -> None:
        # W_2 = floor((W_1 - F + 2P)/S) +1
        super().__init__(*args, **kwargs)
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=96,
            kernel_size=(11, 11),
            stride=(4, 4),
            padding=(0, 0),
        )  # output: (227 - 11)/4 + 1 = 55
        self.pool1: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2)
        )  # output: (55-3)/2 + 1 = 27
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=96,
            out_channels=256,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
        )  # Output: (27-5+4) + 1 = 27
        # reuse pool1, Output: (27 - 3)/2 + 1 = 13
        self.conv3: nn.Conv2d = nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # output: (13-3 +2) + 1 = 13
        self.conv4: nn.Conv2d = nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # output: (13 - 3 +2) + 1 = 13
        self.conv5: nn.Conv2d = nn.Conv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # output: (13 - 3 +2) + 1 = 13

        # pool1, output: (13 - 3)/2 + 1 = 6, spatial dim: (B, 256, 6, 6) = 9216
        self.flatten: nn.Flatten = nn.Flatten(1)

        # reuse pool1
        self.fc1: nn.Linear = nn.Linear(9216, 4096)
        self.dropout: nn.Dropout = nn.Dropout(p=0.5)
        self.fc2: nn.Linear = nn.Linear(4096, 4096)
        self.fc3: nn.Linear = nn.Linear(4096, num_classes)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.flatten(x)

        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x
