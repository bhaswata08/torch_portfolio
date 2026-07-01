from typing import Any, override

import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.layer: nn.Sequential = nn.Sequential(
            # 1x1 conv
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=4 * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),  # unchanged spatial dim
            # 3x3 conv
            nn.BatchNorm2d(num_features=4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=4 * growth_rate,
                out_channels=growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  # (W -3 + 2)/1 + 1 = W, unchanged spatial dim
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layer(x)
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        layers: list[DenseLayer] = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block: nn.Sequential = nn.Sequential(*layers)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransitionLayer(nn.Module):
    """
    The task of transition Layer is to reduce the channel count from Cin to Cin*theta
    and also halves the Spatial dim W -> W/2
    """

    def __init__(
        self, in_channels: int, theta: float, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.layer: nn.Sequential = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(in_channels * theta),
                kernel_size=1,
                stride=1,
            ),  # Cin -> Cin * theta
            nn.AvgPool2d(kernel_size=2, stride=2),  # floor((W-2)/2) + 1 = W/2
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DenseNet121(nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        # Input: 224x224
        self.stem: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),  # (224-7+2*3)/2 = 112
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            ),  # (112 - 3 + 2*1)/2 + 1 = 56
        )
        self.dense: nn.Sequential = nn.Sequential(
            DenseBlock(
                num_layers=6, in_channels=64, growth_rate=32
            ),  # out_channels = in_channels + growth_rate * num_layers, C: 64 + 6 * 32
            TransitionLayer(
                in_channels=64 + 32 * 6, theta=0.5
            ),  # C: 256 * 0.5 = 128, W: 56/2 = 28
            DenseBlock(
                num_layers=12, in_channels=128, growth_rate=32
            ),  # C: 128 + 12 * 32 = 512
            TransitionLayer(
                in_channels=128 + 12 * 32, theta=0.5
            ),  # C: 512 * 0.5 = 256, W: 28/2 = 14
            DenseBlock(
                num_layers=24, in_channels=256, growth_rate=32
            ),  # C: 256 + 24 * 32 = 1024
            TransitionLayer(
                in_channels=256 + 24 * 32, theta=0.5
            ),  # C: 1024 * 0.5 = 512, W: 14/2 = 7
            DenseBlock(
                num_layers=16, in_channels=512, growth_rate=32
            ),  # C: 512 + 16 * 32 = 1024
        )
        self.pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self.flatten: nn.Flatten = nn.Flatten(start_dim=1)
        self.fc1: nn.Linear = nn.Linear(in_features=1024, out_features=num_classes)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.dense(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc1(out)
        return out
