"""Test-set metrics and inference-latency benchmarking."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    accuracy: float
    f1: float


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> EvalResult:
    """Top-1 accuracy and macro F1 of ``model(images)`` over ``loader``."""
    model.eval()
    preds, labels = [], []
    for images, targets in loader:
        preds.append(model(images.to(device)).argmax(dim=1).cpu())
        labels.append(targets)
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    f1 = f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)))
    return EvalResult(accuracy=float((y_pred == y_true).mean()), f1=float(f1))


@torch.no_grad()
def measure_latency(
    model: nn.Module, loader: DataLoader, device: torch.device, warmup_batches: int = 10
) -> dict[str, float]:
    """Time the forward pass of every batch in ``loader`` and return statistics in milliseconds.

    On CUDA, CUDA events are used and the device is synchronised after every batch.
    Per-sample latency is the mean batch latency divided by the size of the first batch.
    """
    model.eval()
    for i, (images, _) in enumerate(loader):
        if i >= warmup_batches:
            break
        model(images.to(device))

    use_cuda = device.type == "cuda"
    timings, batch_size = [], None
    for images, _ in loader:
        images = images.to(device)
        batch_size = batch_size or images.size(0)
        if use_cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(images)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            model(images)
            timings.append((time.perf_counter() - t0) * 1000)

    t = np.asarray(timings)
    per_sample = float(t.mean() / batch_size)
    return {
        "batch_size": batch_size,
        "latency_mean_ms": float(t.mean()),
        "latency_std_ms": float(t.std()),
        "latency_median_ms": float(np.median(t)),
        "latency_min_ms": float(t.min()),
        "latency_max_ms": float(t.max()),
        "latency_per_sample_ms": per_sample,
        "throughput_samples_per_s": 1000.0 / per_sample,
    }
