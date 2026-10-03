from __future__ import annotations

import torch.nn as nn

from vitrm.config import ModelConfig, ResNetConfig, ViTConfig, ViTRMConfig
from vitrm.models.baselines import build_resnet, build_vit
from vitrm.models.vitrm import ViTRM

__all__ = ["ViTRM", "build_model"]


def build_model(cfg: ModelConfig, num_classes: int) -> nn.Module:
    if isinstance(cfg, ViTRMConfig):
        return ViTRM(cfg, num_classes)
    if isinstance(cfg, ViTConfig):
        return build_vit(cfg, num_classes)
    if isinstance(cfg, ResNetConfig):
        return build_resnet(cfg, num_classes)
    raise TypeError(f"Unsupported model config: {type(cfg).__name__}")
