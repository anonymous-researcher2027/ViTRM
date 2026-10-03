from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vitrm.config import ExperimentConfig
from vitrm.models import build_model


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch, and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: Path, model: nn.Module, config: ExperimentConfig, **metadata: Any
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.to_dict(), "model_state": model.state_dict(), **metadata}, path)


def load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[nn.Module, ExperimentConfig, dict]:
    """Rebuild the model stored in ``path``; returns ``(model, config, checkpoint)``."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = ExperimentConfig.from_dict(checkpoint["config"], source=str(path))
    model = build_model(config.model, config.dataset.num_classes)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), config, checkpoint
