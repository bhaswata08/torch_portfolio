from typing import Any, override

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_portfolio.cnn_lineage.models.resnet import ResidualBlock, ResNet


class SeBlock(nn.Module):
    def __init__(self, in_channels: int, r: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pool1: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self.flatten: nn.Flatten = nn.Flatten(1)
        self.fc1: nn.Linear = nn.Linear(in_channels, in_channels // r)
        self.fc2: nn.Linear = nn.Linear(in_channels // r, in_channels)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool1(x)
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = F.sigmoid(out)
        out = out.reshape(out.shape[0], out.shape[1], 1, 1) * x

        return out


class SeResBlock(ResidualBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels, out_channels, stride, *args, **kwargs)
        self.seblock1: SeBlock = SeBlock(in_channels=out_channels, r=16)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.seblock1(out)
        out = F.relu(out + self.shortcut(x))
        return out


class SeResNet(ResNet):
    def __init__(
        self, in_channels: int, num_classes: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(in_channels, num_classes, *args, **kwargs)

        self.reslayer2: nn.Sequential = nn.Sequential(
            SeResBlock(in_channels=64, out_channels=64, stride=1),
            SeResBlock(in_channels=64, out_channels=64, stride=1),
        )

        self.reslayer3: nn.Sequential = nn.Sequential(
            SeResBlock(in_channels=64, out_channels=128, stride=2),
            SeResBlock(in_channels=128, out_channels=128, stride=1),
        )

        self.reslayer4: nn.Sequential = nn.Sequential(
            SeResBlock(in_channels=128, out_channels=256, stride=2),
            SeResBlock(in_channels=256, out_channels=256, stride=1),
        )

        self.reslayer5: nn.Sequential = nn.Sequential(
            SeResBlock(in_channels=256, out_channels=512, stride=2),
            SeResBlock(in_channels=512, out_channels=512, stride=1),
        )
