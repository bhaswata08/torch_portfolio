from typing import Any, cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(
        self, input_size: int, num_classes: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.flatten: nn.Flatten = nn.Flatten(start_dim=1)
        self.fc1: nn.Linear = nn.Linear(input_size, 50)
        self.fc2: nn.Linear = nn.Linear(50, num_classes)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cast(torch.Tensor, self.flatten(x))
        x = cast(torch.Tensor, self.fc1(x))
        x = F.relu(x)
        return cast(torch.Tensor, self.fc2(x))
