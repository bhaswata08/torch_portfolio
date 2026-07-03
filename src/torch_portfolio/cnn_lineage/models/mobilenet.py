from typing import Any

import torch.nn as nn


class DWConv(nn.Module):
    def __init__(self, in_channels: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=32,
            padding=0,
            groups=in_channels,
        )  # 112-


class MobileNetV1(nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1
        )  # (224 - 3 + 2*1)/2 + 1 = 112
        self.conv2dw: nn.Conv2d

        layers = []
        config = [64, 128, 256, 512]
        for i in range(len(config)):
            pass
        layers.extend(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Conv2d(
                    in_channels=32, out_channels=32, kernel_size=3, stride=1, groups=32
                ),
            ]
        )
        # for i in range(7):
        #     (
        #         layers.extend(
        #             [
        #                 nn.Conv2d(
        #                     in_channels=in_channels,
        #                     out_channels=config[i],
        #                     kernel_size=3,
        #                     stride=2,
        #                     padding=1,
        #                 ),
        #                 nn.Conv2d(
        #                     in_channels=
        #                     )
        #             ]
        #         ),
        #     )
