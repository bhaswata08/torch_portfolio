from typing import cast

import hydra
import torch
import torch.nn as nn
import torch.optim as optim

from torch_portfolio.cnn_lineage.config import Config, amp_dtype, device
from torch_portfolio.cnn_lineage.data import (
    DatasetName,
    get_loaders,
    in_channels,
    num_classes,
)
from torch_portfolio.cnn_lineage.models import get_model
from torch_portfolio.cnn_lineage.trainer import evaluate, train_one_epoch


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: Config) -> None:
    dataset = cast(DatasetName, cfg.dataset.name)

    train_loader, test_loader = get_loaders(
        dataset, cfg.train.batch_size, cfg.train.workers
    )
    c = in_channels(dataset)
    n = num_classes(dataset)

    model: nn.Module = get_model(cfg.model.name, c, n).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if cfg.train.compile:
        model = cast(nn.Module, torch.compile(model))  # compile → nn.Module

    print(
        f"Model: {cfg.model.name} | Params: {param_count:,} | Dataset: {cfg.dataset.name}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler(
        device=device.type, enabled=(amp_dtype == torch.float16)
    )

    for epoch in range(1, cfg.train.epochs + 1):
        _ = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch)

    _ = evaluate(model, train_loader, split="train")
    _ = evaluate(model, test_loader, split="test")


if __name__ == "__main__":
    train()
