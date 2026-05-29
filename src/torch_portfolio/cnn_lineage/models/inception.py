from dataclasses import dataclass
from typing import Any, cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InceptionConfig:
    branch1x1: int
    branch3x3: tuple[int, int]
    branch5x5: tuple[int, int]
    branch_proj: int


class AuxClassifier(nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.branch_aux_1: nn.AvgPool2d = nn.AvgPool2d(
            kernel_size=(5, 5), stride=(3, 3)
        )  # (14 - 5)/3 +1 = 4
        self.branch_aux_2: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.branch_aux_3: nn.Flatten = nn.Flatten(1)
        self.branch_aux_4: nn.Linear = nn.Linear(2048, 1024)
        self.branch_aux_5: nn.Dropout = nn.Dropout(0.7)
        self.branch_aux_6: nn.Linear = nn.Linear(1024, num_classes)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.branch_aux_1(x)
        x = F.relu(self.branch_aux_2(x))
        x = self.branch_aux_3(x)
        x = F.relu(self.branch_aux_4(x))
        x = self.branch_aux_5(x)
        x = self.branch_aux_6(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        config: InceptionConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.branch1x1: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.branch1x1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )  # (28-1+0)/1 + 1 = 28
        self.branch3x3_1: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.branch3x3[0],
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )  # (28-1+0)/1 + 1 = 28
        self.branch3x3_2: nn.Conv2d = nn.Conv2d(
            in_channels=config.branch3x3[0],
            out_channels=config.branch3x3[1],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # (28-3+2)/1 + 1 = 28
        self.branch5x5_1: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.branch5x5[0],
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )  # (28-1+0)/1 + 1 = 28
        self.branch5x5_2: nn.Conv2d = nn.Conv2d(
            in_channels=config.branch5x5[0],
            out_channels=config.branch5x5[1],
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
        )  # (28-5+4)/1 + 1 = 28
        self.branch_proj_1: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )  # (28 - 3 + 2)/1 + 1 = 28
        self.branch_proj_2: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.branch_proj,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )  # (28-1+0)/1 + 1 = 28

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = F.relu(self.branch1x1(x))

        b2 = F.relu(self.branch3x3_1(x))
        b2 = F.relu(self.branch3x3_2(b2))

        b3 = F.relu(self.branch5x5_1(x))
        b3 = F.relu(self.branch5x5_2(b3))

        b4 = self.branch_proj_1(x)
        b4 = F.relu(self.branch_proj_2(b4))

        x = torch.cat([b1, b2, b3, b4], dim=1)
        return x


class InceptionNet(nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )  # (224 - 7 + 6)/2 = 112
        self.pool1: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )  # (112 - 3 + 2)/2 + 1 = 56
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )  # (56 - 1)/1 + 1 = 56
        self.conv3: nn.Conv2d = nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # (56 - 3 + 2)/1 + 1 = 56
        self.pool2: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )  # (56 - 3 + 2)/2 + 1 = 28

        config_3a = InceptionConfig(
            branch1x1=64, branch3x3=(96, 128), branch5x5=(16, 32), branch_proj=32
        )
        self.inception3a: InceptionBlock = InceptionBlock(
            in_channels=192, config=config_3a
        )  # (64+128+32+32 = 256) 28x28x256

        config_3b = InceptionConfig(
            branch1x1=128, branch3x3=(128, 192), branch5x5=(32, 96), branch_proj=64
        )
        self.inception3b: InceptionBlock = InceptionBlock(
            in_channels=256, config=config_3b
        )  # (128+192+96+64 = 480) 28x28x480

        self.pool3: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )  # (28 - 3 + 2)/2 + 1 = 14

        config_4a = InceptionConfig(
            branch1x1=192, branch3x3=(96, 208), branch5x5=(16, 48), branch_proj=64
        )
        self.inception4a: InceptionBlock = InceptionBlock(
            in_channels=480, config=config_4a
        )  # (192+208+48+64 = 512) 14x14x512

        config_4b = InceptionConfig(
            branch1x1=160, branch3x3=(112, 224), branch5x5=(24, 64), branch_proj=64
        )
        self.inception4b: InceptionBlock = InceptionBlock(
            in_channels=512, config=config_4b
        )  # (160+224+64+64) = 512, 14x14x512

        self.aux_classifier1: AuxClassifier = AuxClassifier(512, num_classes)

        config_4c = InceptionConfig(
            branch1x1=128, branch3x3=(128, 256), branch5x5=(24, 64), branch_proj=64
        )
        self.inception4c: InceptionBlock = InceptionBlock(
            in_channels=512, config=config_4c
        )  # (128+256+64+64) = 512, 14x14x512

        config_4d = InceptionConfig(
            branch1x1=112, branch3x3=(144, 288), branch5x5=(32, 64), branch_proj=64
        )
        self.inception4d: InceptionBlock = InceptionBlock(
            in_channels=512, config=config_4d
        )  # (112+288+64+64) = 512, 14x14x528

        self.aux_classifier2: AuxClassifier = AuxClassifier(528, num_classes)

        config_4e = InceptionConfig(
            branch1x1=256, branch3x3=(160, 320), branch5x5=(32, 128), branch_proj=128
        )
        self.inception4e: InceptionBlock = InceptionBlock(
            in_channels=528, config=config_4e
        )  # (256+320+128+128) = 832, 14x14x832

        self.pool4: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )  # (14 - 3 + 2)/2 + 1 = 7x7x832

        config_5a = InceptionConfig(
            branch1x1=256, branch3x3=(160, 320), branch5x5=(32, 128), branch_proj=128
        )
        self.inception5a: InceptionBlock = InceptionBlock(
            in_channels=832, config=config_5a
        )  # (256+320+128+128) = 832, 7x7x832

        config_5b = InceptionConfig(
            branch1x1=384, branch3x3=(192, 384), branch5x5=(48, 128), branch_proj=128
        )
        self.inception5b: InceptionBlock = InceptionBlock(
            in_channels=832, config=config_5b
        )  # (384+384+128+128) = 1024, 7x7x1024

        self.pool5: nn.AvgPool2d = nn.AvgPool2d(
            kernel_size=(7, 7), stride=(1, 1)
        )  # (7-7)/1 + 1 = 1x1x1024

        self.flatten: nn.Flatten = nn.Flatten(1)
        self.dropout: nn.Dropout = nn.Dropout(0.4)
        self.fc1: nn.Linear = nn.Linear(1024, num_classes)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        x = self.inception4a(x)
        aux1 = self.aux_classifier1(x) if self.training else None
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux_classifier2(x) if self.training else None
        x = self.inception4e(x)
        x = self.pool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)

        if self.training:
            x = 0.3 * cast(torch.Tensor, aux1) + 0.3 * cast(torch.Tensor, aux2) + x

        return cast(torch.Tensor, x)
