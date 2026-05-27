from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch_portfolio.cnn_lineage.config import amp_dtype, device

type Loader = DataLoader[tuple[torch.Tensor, torch.Tensor]]


def train_one_epoch(
    model: nn.Module,
    loader: Loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
) -> float:
    """Runs one full pass over train_loader. Returns mean loss for the epoch."""
    # Assign to _ to satisfy strict "unused return value" checks
    _ = model.train()
    total_loss = 0.0

    for batch in loader:
        data = cast(torch.Tensor, batch[0]).to(device, non_blocking=True)
        target = cast(torch.Tensor, batch[1]).to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(device.type != "cpu"),
        ):
            # nn.Module call returns Any; cast to Tensor
            scores = cast(torch.Tensor, model(data))
            loss = cast(torch.Tensor, criterion(scores, target))

        optimizer.zero_grad(set_to_none=True)

        scaled_loss = scaler.scale(loss)
        _ = scaled_loss.backward()

        _ = scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())

    mean_loss = total_loss / len(loader)
    print(f"Epoch {epoch:>3} | loss: {mean_loss:.4f}")
    return mean_loss


@torch.no_grad()
def evaluate(model: nn.Module, loader: Loader, split: str = "test") -> float:
    """Returns accuracy on the given loader."""
    _ = model.eval()
    correct = 0
    total = 0

    for batch in loader:
        x = cast(torch.Tensor, batch[0]).to(device, non_blocking=True)
        y = cast(torch.Tensor, batch[1]).to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(device.type != "cpu"),
        ):
            scores = cast(torch.Tensor, model(x))

        _, preds = cast(tuple[torch.Tensor, torch.Tensor], scores.max(1))

        matches = preds == y
        correct += int(matches.sum().item())
        total += int(y.size(0))

    acc = correct / total
    print(f"{split:>5} accuracy: {correct}/{total} ({acc * 100:.2f}%)")

    _ = model.train()
    return acc
