# Imports
import os
from typing import Any, cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

type Loader = DataLoader[tuple[torch.Tensor, torch.Tensor]]
_ = torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    device = torch.device("cpu")
    amp_dtype = torch.float32


# FCN
class NN(nn.Module):
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
        fc1_out: torch.Tensor = cast(torch.Tensor, self.fc1(x))
        x = F.relu(fc1_out)
        return cast(torch.Tensor, self.fc2(x))


# CNN
class CNN(nn.Module):
    def __init__(
        self, in_channel: int, num_classes: int = 10, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channel,
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
        x = F.relu(x)
        x = cast(torch.Tensor, self.pool(x))
        x = F.relu(self.conv2(x))
        x = cast(torch.Tensor, self.pool(x))
        x = x.flatten(1)
        x = self.fc1(x)

        return x


if __name__ == "__main__":
    # Hyperparams
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 10

    train_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.AutoAugment(),
        ]
    )

    test_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    # Load data
    train_dataset = datasets.MNIST(
        root="data/MNIST", train=True, transform=train_transforms, download=True
    )
    train_loader: Loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=cast(int, os.cpu_count()),
        pin_memory=True if device.type == "cuda" else False,
    )

    test_dataset = datasets.MNIST(
        root="data/MNIST", train=False, transform=test_transforms, download=True
    )
    test_loader: Loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=cast(int, os.cpu_count()),
        pin_memory=True if device.type == "cuda" else False,
    )

    # Initialize network
    # model = NN(784, num_classes).to(device)
    model = CNN(1, 10).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    scaler = torch.amp.GradScaler(
        device=device.type, enabled=(amp_dtype == torch.float16)
    )

    model = torch.compile(model)

    # Train network
    for epoch in range(num_epochs):
        for idx, batch in enumerate(train_loader):
            data, target = batch

            data: torch.Tensor = data.to(device=device, non_blocking=True)
            target: torch.Tensor = target.to(device=device, non_blocking=True)

            # Forward
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(device.type != "cpu")
            ):
                scores = cast(torch.Tensor, model(data))
                loss = cast(torch.Tensor, criterion(scores, target))

            optimizer.zero_grad(set_to_none=True)

            # Backward
            _ = scaler.scale(loss).backward()
            _ = scaler.step(optimizer)
            scaler.update()

    # Check accuracy
    def check_accuracy(loader: Loader, model: nn.Module | Any):
        if loader.dataset.train:
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")

        num_correct = 0
        num_samples = 0
        _ = model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device, non_blocking=True)
                y = y.to(device=device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=(device.type != "cpu"),
                ):
                    scores = cast(torch.Tensor, model(x))

                _, preds = scores.max(1)
                num_correct += (preds == y).sum().item()
                num_samples += preds.size(0)

            print(
                f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
            )
        acc: float = float(num_correct) / float(num_samples)
        _ = model.train()  # In case you want to keep training after checking accuracy
        return acc

    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)
