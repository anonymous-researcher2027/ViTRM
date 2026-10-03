"""Command-line interface: ``vitrm train | grid-search | evaluate | download`` (or ``python -m vitrm``)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import click

from vitrm.config import ExperimentConfig
from vitrm.utils import setup_logging


def _load_config(path: str, data_root: str | None, epochs: int | None) -> ExperimentConfig:
    config = ExperimentConfig.from_yaml(path)
    if data_root is not None:
        config.dataset = dataclasses.replace(config.dataset, root=data_root)
    if epochs is not None:
        config.training = dataclasses.replace(config.training, epochs=epochs)
    return config


def _train_options(fn):
    """Options shared by ``train`` and ``grid-search``."""
    options = [
        click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False), help="Experiment YAML file."),
        click.option("--run-name", default=None, help="Run name (default: derived from the config path plus a timestamp)."),
        click.option("--output-dir", default="outputs", show_default=True, type=click.Path(file_okay=False), help="Where run directories are written."),
        click.option("--experiment-name", default=None, help="MLflow experiment (default: '<dataset>-classification')."),
        click.option("--data-root", default=None, help="Override dataset.root (CIFAR is downloaded there if missing)."),
        click.option("--epochs", type=int, default=None, help="Override training.epochs."),
        click.option("--device", default="auto", show_default=True, help="'auto' (GPU if available), 'cuda' or 'cpu'."),
        click.option("--num-workers", type=int, default=4, show_default=True, help="DataLoader workers."),
        click.option("--max-batches", type=int, default=None, help="Stop each epoch after this many batches (debugging)."),
        click.option("--mlflow/--no-mlflow", "tracking", default=True, show_default=True, help="Log the run to MLflow."),
    ]  # fmt: skip
    for option in reversed(options):
        fn = option(fn)
    return fn


@click.group()
def main() -> None:
    """Train and evaluate ViTRM and the ViT / ResNet baselines on CIFAR-10/100."""
    setup_logging()


@main.command()
@_train_options
def train(config_path, data_root, epochs, **kwargs) -> None:
    """Train one model from a config file."""
    from vitrm.training import train as run_training

    run_training(_load_config(config_path, data_root, epochs), **kwargs)


@main.command("grid-search")
@_train_options
@click.option("--n-latent-steps", "n_latent_steps", type=int, multiple=True, default=(1, 2, 3, 6, 12), show_default=True, help="Values of n (repeat the option).")  # fmt: skip
@click.option("--n-supervision", "n_supervision", type=int, multiple=True, default=(1, 2, 4, 8, 16), show_default=True, help="Values of N_sup (repeat the option).")  # fmt: skip
def grid_search(config_path, data_root, epochs, **kwargs) -> None:
    """Train a ViTRM for every (n_latent_steps, n_supervision) pair."""
    from vitrm.training import grid_search as run_grid_search

    results = run_grid_search(_load_config(config_path, data_root, epochs), **kwargs)
    click.echo(json.dumps(results, indent=2))


@main.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False), help="A best.pt file written by 'vitrm train'.")  # fmt: skip
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option(
    "--data-root", default=None, help="Override the dataset root stored in the checkpoint."
)
@click.option("--device", default="auto", show_default=True)
@click.option("--num-workers", type=int, default=4, show_default=True)
@click.option("--benchmark", is_flag=True, help="Also measure inference latency and throughput.")
@click.option(
    "--output", type=click.Path(dir_okay=False), default=None, help="Write the results as JSON."
)
def evaluate(checkpoint, batch_size, data_root, device, num_workers, benchmark, output) -> None:
    """Evaluate a trained checkpoint on the test set."""
    from vitrm.data import build_loaders
    from vitrm.evaluation import evaluate as run_evaluation
    from vitrm.evaluation import measure_latency
    from vitrm.utils import count_parameters, load_checkpoint, resolve_device

    torch_device = resolve_device(device)
    model, config, _ = load_checkpoint(checkpoint, torch_device)
    if data_root is not None:
        config.dataset = dataclasses.replace(config.dataset, root=data_root)
    _, test_loader = build_loaders(config.dataset, batch_size, num_workers, randaugment=False)

    result = run_evaluation(model, test_loader, torch_device, config.dataset.num_classes)
    results = {
        "checkpoint": str(checkpoint),
        "dataset": config.dataset.name,
        "num_parameters": count_parameters(model),
        "test_accuracy": result.accuracy,
        "test_f1": result.f1,
    }
    if benchmark:
        results.update(
            device=str(torch_device), **measure_latency(model, test_loader, torch_device)
        )

    click.echo(json.dumps(results, indent=2))
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(results, indent=2))


@main.command()
@click.option("--data-root", default="./data", show_default=True, help="Download directory.")
@click.option(
    "--dataset",
    "names",
    type=click.Choice(["cifar10", "cifar100"]),
    multiple=True,
    default=("cifar10", "cifar100"),
    show_default=True,
)
def download(data_root, names) -> None:
    """Download CIFAR-10/100 ahead of training."""
    from vitrm.data import DATASETS

    for name in names:
        for train in (True, False):
            DATASETS[name](data_root, train=train, download=True)
