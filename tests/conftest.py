from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from vitrm.config import ExperimentConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def tiny_config(family: str = "vitrm", **model_overrides) -> ExperimentConfig:
    models = {
        "vitrm": dict(
            family="vitrm", dim=32, num_heads=2, mlp_hidden_dim=32,
            n_latent_steps=2, t_deep=2, n_supervision=2, latent_dropout=0.1, token_dropout=0.1,
        ),
        "vit": dict(family="vit", dim=32, depth=2, heads=2, mlp_dim=32),
        "resnet": dict(family="resnet", depth=18),
    }  # fmt: skip
    return ExperimentConfig.from_dict(
        {
            "dataset": {"name": "cifar10"},
            "model": {**models[family], **model_overrides},
            "training": {
                "epochs": 2,
                "batch_size": 8,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "warmup_epochs": 1,
                "ema_decay": 0.9,
            },
        }  # fmt: skip
    )


def synthetic_loader(
    num_samples: int = 16, batch_size: int = 8, num_classes: int = 10
) -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(num_samples, 3, 32, 32, generator=generator)
    labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


@pytest.fixture
def loaders(monkeypatch):
    """Replace the CIFAR loaders used by the trainer and CLI with small synthetic ones."""

    def fake_build_loaders(cfg, batch_size, num_workers=0, randaugment=True):
        return synthetic_loader(batch_size=batch_size), synthetic_loader(batch_size=batch_size)

    monkeypatch.setattr("vitrm.training.trainer.build_loaders", fake_build_loaders)
    monkeypatch.setattr("vitrm.data.build_loaders", fake_build_loaders)
    return fake_build_loaders
