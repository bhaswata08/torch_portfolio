from typing import Any, override

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGNet(nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # (224 - 3 + 2) + 1 = 224
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # (224 - 3 + 2) + 1 = 224
        self.pool1: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2)
        )  # (224 - 2)/2 + 1 = 112
        self.conv3: nn.Conv2d = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # (112 - 3 + 2) + 1 = 112
        self.conv4: nn.Conv2d = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # (112 - 3 + 2) + 1 = 112
        # self.pool1, (112 -2)/2 + 1 = 56
        self.conv5: nn.Conv2d = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # (56 - 3 + 2) + 1 = 56
        self.conv6: nn.Conv2d = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # (56 - 3 + 2) + 1 = 56
        self.conv7: nn.Conv2d = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # (56 - 3 + 2) + 1 = 56
        # self.pool1, (56-2)/2 + 1 = 28
        self.conv8: nn.Conv2d = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # (28 - 3 + 2) + 1 = 28
        self.conv9: nn.Conv2d = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # (28 - 3 + 2) + 1 = 28
        self.conv10: nn.Conv2d = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # (28 - 3 + 2) + 1 = 28
        # self.pool1, (28-2)/2 + 1 = 14
        self.conv11: nn.Conv2d = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # (14 - 3 + 2) + 1 = 14
        self.conv12: nn.Conv2d = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # (14 - 3 + 2) + 1 = 14
        self.conv13: nn.Conv2d = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # (14 - 3 + 2) + 1 = 14
        # self.pool1, (14-2)/2 + 1 = 7
        self.flatten: nn.Flatten = nn.Flatten(1)
        self.dropout: nn.Dropout = nn.Dropout(0.5)
        self.fc1: nn.Linear = nn.Linear(7 * 7 * 512, 4096)
        self.fc2: nn.Linear = nn.Linear(4096, 4096)
        self.fc3: nn.Linear = nn.Linear(4096, num_classes)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv11(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = F.relu(x)
        x = self.conv13(x)
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
