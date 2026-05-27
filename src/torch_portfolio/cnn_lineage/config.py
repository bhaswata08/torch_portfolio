from dataclasses import dataclass, field

import torch
from hydra.core.config_store import ConfigStore

# ── Device & AMP (unchanged) ──────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    device = torch.device("cpu")
    amp_dtype = torch.float32

_ = torch.set_float32_matmul_precision("high")


# ── Structured configs (schema + type safety) ─────────────────────────────────
@dataclass
class ModelConfig:
    name: str = "cnn"


@dataclass
class DatasetConfig:
    name: str = "mnist"


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    compile: bool = True
    workers: int | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Register with Hydra's config store
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
