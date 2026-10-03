"""Per-epoch training loops: a standard loop for baselines and a deep-supervision loop for ViTRM."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from vitrm.models.vitrm import ViTRM
from vitrm.training.augment import MixedBatch, mixed_cross_entropy
from vitrm.training.ema import EMA

logger = logging.getLogger(__name__)

BatchAugment = Callable[[torch.Tensor, torch.Tensor], MixedBatch]
LOG_EVERY = 50


@dataclass
class EpochStats:
    loss: float
    accuracy: float
    f1: float


class _Meter:
    """Accumulates the per-sample loss and predictions over an epoch."""

    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.count = 0
        self.preds: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

    def update(self, loss: float, preds: torch.Tensor, labels: torch.Tensor) -> None:
        self.loss_sum += loss * labels.size(0)
        self.count += labels.size(0)
        self.preds.append(preds.cpu())
        self.labels.append(labels.cpu())

    def stats(self, num_classes: int) -> EpochStats:
        preds = torch.cat(self.preds).numpy()
        labels = torch.cat(self.labels).numpy()
        accuracy = float((preds == labels).mean())
        f1 = f1_score(labels, preds, average="macro", labels=list(range(num_classes)))
        return EpochStats(self.loss_sum / max(self.count, 1), accuracy, float(f1))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    augment: BatchAugment,
    ema: EMA | None = None,
    max_batches: int | None = None,
) -> EpochStats:
    """Standard supervised epoch (ViT and ResNet baselines).

    Training accuracy and F1 are measured against the original (un-mixed) labels.
    """
    model.train()
    meter = _Meter()
    for step, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        batch = augment(images, labels)

        logits = model(batch.images)
        loss = mixed_cross_entropy(logits, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)

        meter.update(loss.item(), logits.argmax(dim=1), labels)
        if step % LOG_EVERY == 0:
            logger.info("  [batch %d] loss=%.4f", step, loss.item())
        if max_batches is not None and step >= max_batches:
            break
    return meter.stats(num_classes)


def train_one_epoch_deep_supervision(
    model: ViTRM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    augment: BatchAugment,
    ema: EMA | None = None,
    max_batches: int | None = None,
) -> EpochStats:
    """ViTRM epoch with deep supervision.

    Each batch runs up to ``n_supervision`` recursion steps. Every step takes its own
    optimizer step on ``cross_entropy + BCE(halt_logit, prediction_is_correct)``, and the
    (y, z) state is detached between steps. The loop stops early for the batch once the
    mean halting probability exceeds ``halt_threshold``. The reported loss is the sum of
    the step losses.
    """
    cfg = model.cfg
    model.train()
    meter = _Meter()
    for step, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        batch = augment(images, labels)

        y = z = None
        batch_loss = 0.0
        for _ in range(cfg.n_supervision):
            if y is not None:
                y, z = y.detach(), z.detach()
            y, z, logits, halt_logit = model.deep_recursion(batch.images, y, z)

            preds = logits.argmax(dim=1)
            correct = (preds == labels).float()
            loss = mixed_cross_entropy(logits, batch) + F.binary_cross_entropy_with_logits(
                halt_logit, correct
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)

            batch_loss += loss.item()
            if torch.sigmoid(halt_logit).mean().item() > cfg.halt_threshold:
                break

        meter.update(batch_loss, preds, labels)
        if step % LOG_EVERY == 0:
            logger.info("  [batch %d] loss=%.4f", step, batch_loss)
        if max_batches is not None and step >= max_batches:
            break
    return meter.stats(num_classes)
