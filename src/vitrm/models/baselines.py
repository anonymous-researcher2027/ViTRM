"""Baseline architectures: ViT (via ``vit-pytorch``) and torchvision ResNets trained from scratch."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from vit_pytorch import ViT

from vitrm.config import ResNetConfig, ViTConfig

RESNETS = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50}


def build_vit(cfg: ViTConfig, num_classes: int) -> nn.Module:
    return ViT(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        num_classes=num_classes,
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        mlp_dim=cfg.mlp_dim,
        dropout=cfg.dropout,
        emb_dropout=cfg.emb_dropout,
    )


def build_resnet(cfg: ResNetConfig, num_classes: int) -> nn.Module:
    if cfg.depth not in RESNETS:
        raise ValueError(f"Unsupported ResNet depth {cfg.depth}; expected one of {list(RESNETS)}")
    return RESNETS[cfg.depth](weights=None, num_classes=num_classes)
