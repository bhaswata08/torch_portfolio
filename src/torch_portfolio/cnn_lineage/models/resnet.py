from typing import Any, override

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )  # x1 = (W - 2)/S + 1 # x1 x x1 x Co
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(num_features=out_channels)
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # x2 = (x1 - 3 + 2*1) + 1 = x1  = (W-2)/S # x2 x x2 x Co

        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(num_features=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Sequential | nn.Identity = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                ),  # x0 = (W-1)/S + 1, x0 == x2
                nn.BatchNorm2d(num_features=out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class ResNet(nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        # Input: 224x224
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )  # (224 - 7 +2*3)/2 + 1 = 112
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(num_features=64)
        self.pool1: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )  # (112 - 3 + 2*1)/2 + 1 = 56

        self.reslayer2: nn.Sequential = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=64, stride=1),
            ResidualBlock(in_channels=64, out_channels=64, stride=1),
        )

        self.reslayer3: nn.Sequential = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=128, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, stride=1),
        )

        self.reslayer4: nn.Sequential = nn.Sequential(
            ResidualBlock(in_channels=128, out_channels=256, stride=2),
            ResidualBlock(in_channels=256, out_channels=256, stride=1),
        )

        self.reslayer5: nn.Sequential = nn.Sequential(
            ResidualBlock(in_channels=256, out_channels=512, stride=2),
            ResidualBlock(in_channels=512, out_channels=512, stride=1),
        )

        self.pool2: nn.AvgPool2d = nn.AvgPool2d(
            kernel_size=7, stride=1, padding=0
        )  # (7 - 7)/1 + 1 = 1

        self.flatten: nn.Flatten = nn.Flatten(1)  # 512 x 1 x 1 -> 512

        self.fc1: nn.Linear = nn.Linear(in_features=512, out_features=num_classes)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool1(out)
        out = self.reslayer2(out)
        out = self.reslayer3(out)
        out = self.reslayer4(out)
        out = self.reslayer5(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        return out
