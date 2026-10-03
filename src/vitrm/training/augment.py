"""Batch-level Mixup / CutMix augmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MixedBatch:
    images: torch.Tensor
    labels_a: torch.Tensor
    labels_b: torch.Tensor
    lam: float


def mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float) -> MixedBatch:
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[index]
    return MixedBatch(mixed, labels, labels[index], lam)


def cutmix(images: torch.Tensor, labels: torch.Tensor, alpha: float) -> MixedBatch:
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    height, width = images.shape[-2:]

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
    cy, cx = np.random.randint(height), np.random.randint(width)
    y1, y2 = np.clip(cy - cut_h // 2, 0, height), np.clip(cy + cut_h // 2, 0, height)
    x1, x2 = np.clip(cx - cut_w // 2, 0, width), np.clip(cx + cut_w // 2, 0, width)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    # Correct lambda for the clipped box so it equals the exact pixel ratio kept from the original.
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (height * width)
    return MixedBatch(mixed, labels, labels[index], float(lam))


class MixupCutmix:
    """Apply Mixup or CutMix to each batch, choosing between them with equal probability."""

    def __init__(self, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0) -> None:
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> MixedBatch:
        if np.random.rand() < 0.5:
            return mixup(images, labels, self.mixup_alpha)
        return cutmix(images, labels, self.cutmix_alpha)


def identity(images: torch.Tensor, labels: torch.Tensor) -> MixedBatch:
    return MixedBatch(images, labels, labels, 1.0)


def mixed_cross_entropy(logits: torch.Tensor, batch: MixedBatch) -> torch.Tensor:
    if batch.lam == 1.0:
        return F.cross_entropy(logits, batch.labels_a)
    return batch.lam * F.cross_entropy(logits, batch.labels_a) + (1 - batch.lam) * F.cross_entropy(
        logits, batch.labels_b
    )
