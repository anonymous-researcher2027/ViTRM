from __future__ import annotations

import json

import pytest
import torch
from click.testing import CliRunner

from conftest import synthetic_loader, tiny_config
from vitrm.cli import main
from vitrm.models import build_model
from vitrm.training import grid_search, train
from vitrm.training.augment import MixupCutmix, cutmix, identity, mixed_cross_entropy, mixup
from vitrm.training.ema import EMA
from vitrm.training.engine import train_one_epoch, train_one_epoch_deep_supervision
from vitrm.training.optim import build_optimizer, build_scheduler
from vitrm.utils import load_checkpoint, set_seed


def test_mixup_and_cutmix_preserve_shapes_and_lambda_range():
    set_seed(0)
    images, labels = torch.randn(8, 3, 32, 32), torch.arange(8)
    for fn, alpha in ((mixup, 0.2), (cutmix, 1.0)):
        batch = fn(images, labels, alpha)
        assert batch.images.shape == images.shape
        assert 0.0 <= batch.lam <= 1.0
        assert sorted(batch.labels_b.tolist()) == labels.tolist()


def test_cutmix_does_not_modify_its_input():
    images = torch.randn(4, 3, 32, 32)
    original = images.clone()
    cutmix(images, torch.arange(4), alpha=1.0)
    torch.testing.assert_close(images, original)


def test_mixed_cross_entropy_without_mixing_is_plain_cross_entropy():
    logits, labels = torch.randn(4, 10), torch.arange(4)
    expected = torch.nn.functional.cross_entropy(logits, labels)
    torch.testing.assert_close(mixed_cross_entropy(logits, identity(logits, labels)), expected)


def test_warmup_then_cosine_schedule():
    cfg = tiny_config().training
    cfg.epochs, cfg.warmup_epochs = 10, 4
    model = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    lrs = []
    for _ in range(cfg.epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == 0.0
    assert lrs[cfg.warmup_epochs] == pytest.approx(cfg.learning_rate)
    assert lrs[cfg.warmup_epochs :] == sorted(lrs[cfg.warmup_epochs :], reverse=True)


def test_ema_tracks_the_model():
    model = torch.nn.Linear(2, 2)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.add_(1.0)
    before = ema.module.weight.clone()
    ema.update(model)
    torch.testing.assert_close(ema.module.weight, 0.5 * before + 0.5 * model.weight)


@pytest.mark.parametrize("family", ["vitrm", "vit"])
def test_one_epoch_updates_weights_with_finite_loss(family):
    set_seed(0)
    config = tiny_config(family)
    model = build_model(config.model, num_classes=10)
    optimizer = build_optimizer(model, config.training)
    epoch_fn = train_one_epoch_deep_supervision if family == "vitrm" else train_one_epoch
    before = [p.detach().clone() for p in model.parameters()]
    stats = epoch_fn(
        model, synthetic_loader(), optimizer, torch.device("cpu"), 10, MixupCutmix(), EMA(model)
    )
    assert torch.isfinite(torch.tensor(stats.loss))
    assert 0.0 <= stats.accuracy <= 1.0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_train_writes_outputs_and_checkpoint_reloads(tmp_path, loaders):
    summary = train(
        tiny_config(), run_name="smoke", output_dir=tmp_path, num_workers=0, tracking=False
    )
    run_dir = tmp_path / "smoke"
    for name in ("config.yaml", "history.json", "summary.json", "best.pt"):
        assert (run_dir / name).exists()
    assert len(json.loads((run_dir / "history.json").read_text())) == 2
    assert 1 <= summary["best_epoch"] <= 2

    model, config, checkpoint = load_checkpoint(run_dir / "best.pt", torch.device("cpu"))
    assert config.model == tiny_config().model
    assert checkpoint["epoch"] == summary["best_epoch"]


def test_grid_search_runs_every_pair(tmp_path, loaders):
    config = tiny_config()
    config.training.epochs = 1
    results = grid_search(
        config, n_latent_steps=[1, 2], n_supervision=[1, 2], run_name="grid",
        output_dir=tmp_path, num_workers=0, tracking=False,
    )  # fmt: skip
    assert [(r["n_latent_steps"], r["n_supervision"]) for r in results] == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]
    assert (tmp_path / "grid" / "n2-nsup2" / "best.pt").exists()


def test_cli_train_and_evaluate(tmp_path, loaders):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(tiny_config("resnet").to_dict()))  # JSON is valid YAML
    runner = CliRunner()
    result = runner.invoke(main, [
        "train", "--config", str(config_path), "--run-name", "cli", "--output-dir", str(tmp_path),
        "--epochs", "1", "--num-workers", "0", "--no-mlflow",
    ])  # fmt: skip
    assert result.exit_code == 0, result.output

    out = tmp_path / "eval.json"
    result = runner.invoke(main, [
        "evaluate", "--checkpoint", str(tmp_path / "cli" / "best.pt"), "--batch-size", "8",
        "--num-workers", "0", "--benchmark", "--output", str(out),
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    metrics = json.loads(out.read_text())
    assert {
        "test_accuracy",
        "test_f1",
        "latency_mean_ms",
        "throughput_samples_per_s",
    } <= metrics.keys()
