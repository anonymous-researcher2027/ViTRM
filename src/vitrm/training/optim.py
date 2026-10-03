from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler, SequentialLR

from vitrm.config import TrainingConfig


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=cfg.betas
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainingConfig) -> LRScheduler:
    """Per-epoch schedule: linear warmup from 0 for ``warmup_epochs``, then cosine decay to ``min_lr``."""
    if cfg.warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
    warmup = LambdaLR(optimizer, lambda epoch: min(1.0, epoch / cfg.warmup_epochs))
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs - cfg.warmup_epochs), eta_min=cfg.min_lr
    )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.warmup_epochs])
