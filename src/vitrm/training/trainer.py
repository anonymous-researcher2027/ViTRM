"""End-to-end training of a single experiment, and the ViTRM recursion-depth grid search."""

from __future__ import annotations

import dataclasses
import itertools
import json
import logging
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from vitrm.config import ExperimentConfig, ViTRMConfig, flatten
from vitrm.data import build_loaders
from vitrm.evaluation import evaluate
from vitrm.models import ViTRM, build_model
from vitrm.tracking import MlflowTracker, Tracker
from vitrm.training.augment import MixupCutmix, identity
from vitrm.training.ema import EMA
from vitrm.training.engine import train_one_epoch, train_one_epoch_deep_supervision
from vitrm.training.optim import build_optimizer, build_scheduler
from vitrm.utils import count_parameters, resolve_device, save_checkpoint, set_seed

logger = logging.getLogger(__name__)


def default_run_name(config: ExperimentConfig) -> str:
    """``<model>-<dataset>-<config name>-<timestamp>``, derived from the config file path."""
    stem = "-".join(Path(config.source).with_suffix("").parts[-3:]) if config.source else "run"
    return f"{stem}-{datetime.now():%Y%m%d-%H%M%S}"


def train(
    config: ExperimentConfig,
    run_name: str | None = None,
    output_dir: str | Path = "outputs",
    experiment_name: str | None = None,
    device: str = "auto",
    num_workers: int = 4,
    max_batches: int | None = None,
    tracking: bool = True,
) -> dict[str, Any]:
    """Train ``config`` and return a summary of the best test metrics.

    The run directory ``<output_dir>/<run_name>`` receives ``config.yaml``, ``history.json``,
    ``summary.json`` and ``best.pt`` (the weights with the best test accuracy; these are the
    EMA weights when ``training.ema_decay`` is set, since those are the ones evaluated).
    """
    tcfg = config.training
    set_seed(tcfg.seed)
    torch_device = resolve_device(device)
    num_classes = config.dataset.num_classes
    run_name = run_name or default_run_name(config)
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))

    train_loader, test_loader = build_loaders(
        config.dataset, tcfg.batch_size, num_workers, randaugment=tcfg.randaugment
    )
    model = build_model(config.model, num_classes).to(torch_device)
    optimizer = build_optimizer(model, tcfg)
    scheduler = build_scheduler(optimizer, tcfg)
    ema = EMA(model, tcfg.ema_decay) if tcfg.ema_decay else None
    eval_model = ema.module if ema else model
    augment = MixupCutmix(tcfg.mixup_alpha, tcfg.cutmix_alpha) if tcfg.mixup_cutmix else identity
    run_epoch = train_one_epoch_deep_supervision if isinstance(model, ViTRM) else train_one_epoch

    num_params = count_parameters(model)
    logger.info(
        "Run %s | %s | %d parameters | device %s", run_name, config.source, num_params, torch_device
    )

    tracker = (
        MlflowTracker(experiment_name or f"{config.dataset.name}-classification", run_name)
        if tracking
        else Tracker()
    )
    history: list[dict[str, float]] = []
    best = {"best_test_accuracy": 0.0, "best_test_f1": 0.0, "best_epoch": 0}
    with tracker:
        tracker.log_params(
            {**flatten(config.to_dict()), "num_parameters": num_params, "device": str(torch_device)}
        )
        for epoch in range(1, tcfg.epochs + 1):
            lr = optimizer.param_groups[0]["lr"]
            start = time.time()
            train_stats = run_epoch(
                model, train_loader, optimizer, torch_device, num_classes, augment, ema, max_batches
            )
            epoch_time = time.time() - start
            test = evaluate(eval_model, test_loader, torch_device, num_classes)
            scheduler.step()

            metrics = {
                "train_loss": train_stats.loss,
                "train_accuracy": train_stats.accuracy,
                "train_f1": train_stats.f1,
                "test_accuracy": test.accuracy,
                "test_f1": test.f1,
                "learning_rate": lr,
                "epoch_time": epoch_time,
            }
            tracker.log_metrics(metrics, step=epoch)
            history.append({"epoch": epoch, **metrics})
            logger.info(
                "Epoch %d/%d | lr %.2e | train loss %.4f acc %.2f%% | test acc %.2f%% f1 %.4f | %.0fs",
                epoch, tcfg.epochs, lr, train_stats.loss, 100 * train_stats.accuracy,
                100 * test.accuracy, test.f1, epoch_time,
            )  # fmt: skip

            best["best_test_f1"] = max(best["best_test_f1"], test.f1)
            if test.accuracy > best["best_test_accuracy"]:
                best.update(best_test_accuracy=test.accuracy, best_epoch=epoch)
                save_checkpoint(
                    run_dir / "best.pt", eval_model, config,
                    epoch=epoch, test_accuracy=test.accuracy, test_f1=test.f1,
                )  # fmt: skip

        summary = {"run_name": run_name, "num_parameters": num_params, **best}
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        tracker.log_metrics(best)
        for name in ("config.yaml", "history.json", "summary.json"):
            tracker.log_artifact(str(run_dir / name))

    logger.info(
        "Best test accuracy %.2f%% (epoch %d), best test F1 %.4f. Outputs in %s",
        100 * best["best_test_accuracy"], best["best_epoch"], best["best_test_f1"], run_dir,
    )  # fmt: skip
    return summary


def grid_search(
    config: ExperimentConfig,
    n_latent_steps: Sequence[int],
    n_supervision: Sequence[int],
    run_name: str | None = None,
    **train_kwargs: Any,
) -> list[dict[str, Any]]:
    """Train one ViTRM per ``(n_latent_steps, n_supervision)`` pair; other settings come from ``config``."""
    if not isinstance(config.model, ViTRMConfig):
        raise ValueError("Grid search requires a ViTRM config (model.family: vitrm)")
    base_name = run_name or default_run_name(config)
    results = []
    for n, n_sup in itertools.product(n_latent_steps, n_supervision):
        trial = dataclasses.replace(
            config, model=dataclasses.replace(config.model, n_latent_steps=n, n_supervision=n_sup)
        )
        logger.info("Grid search trial: n_latent_steps=%d, n_supervision=%d", n, n_sup)
        summary = train(trial, run_name=f"{base_name}/n{n}-nsup{n_sup}", **train_kwargs)
        results.append({"n_latent_steps": n, "n_supervision": n_sup, **summary})
    return results
