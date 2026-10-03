"""Typed experiment configuration loaded from YAML files.

A config file has three sections::

    dataset:  {name, root}
    model:    {family, ...family-specific fields}
    training: {epochs, batch_size, learning_rate, ...}

``model.family`` selects the model dataclass explicitly (``vitrm``, ``vit`` or
``resnet``). Unknown keys raise an error instead of being silently ignored.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import yaml

NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


@dataclass
class DatasetConfig:
    name: str
    root: str = "./data"

    def __post_init__(self) -> None:
        if self.name not in NUM_CLASSES:
            raise ValueError(f"Unknown dataset {self.name!r}; expected one of {list(NUM_CLASSES)}")

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES[self.name]


@dataclass
class ViTRMConfig:
    """ViTRM hyperparameters.

    Recursion parameters map to the paper's notation as follows:
    ``n_latent_steps`` = n (latent updates of z per recursion step),
    ``t_deep`` = T (recursion steps chained at inference),
    ``n_supervision`` = N_sup (deep-supervision steps per training batch).
    """

    family: str = "vitrm"
    image_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    dim: int = 192
    num_heads: int = 3
    mlp_hidden_dim: int = 384
    n_latent_steps: int = 2
    t_deep: int = 3
    n_supervision: int = 2
    halt_threshold: float = 0.5
    dropout: float = 0.1
    latent_dropout: float = 0.0
    token_dropout: float = 0.0


@dataclass
class ViTConfig:
    family: str = "vit"
    image_size: int = 32
    patch_size: int = 4
    dim: int = 512
    depth: int = 8
    heads: int = 8
    mlp_dim: int = 1024
    dropout: float = 0.1
    emb_dropout: float = 0.0


@dataclass
class ResNetConfig:
    family: str = "resnet"
    depth: int = 18  # 18, 34 or 50


ModelConfig = Union[ViTRMConfig, ViTConfig, ResNetConfig]
MODEL_CONFIGS: dict[str, type] = {"vitrm": ViTRMConfig, "vit": ViTConfig, "resnet": ResNetConfig}


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    ema_decay: float | None = None  # EMA of weights used for evaluation; None disables it
    randaugment: bool = True
    mixup_cutmix: bool = True  # pick Mixup or CutMix at random for each batch
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        # YAML parses "3e-4" as a string, so coerce numeric fields explicitly.
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.min_lr = float(self.min_lr)
        self.betas = tuple(float(b) for b in self.betas)


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    source: str | None = field(default=None, compare=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: str | None = None) -> ExperimentConfig:
        missing = {"dataset", "model", "training"} - raw.keys()
        if missing:
            raise ValueError(f"Config is missing sections: {sorted(missing)}")
        model_raw = dict(raw["model"])
        family = model_raw.get("family")
        if family not in MODEL_CONFIGS:
            raise ValueError(f"model.family must be one of {list(MODEL_CONFIGS)}, got {family!r}")
        return cls(
            dataset=_build(DatasetConfig, raw["dataset"], "dataset"),
            model=_build(MODEL_CONFIGS[family], model_raw, "model"),
            training=_build(TrainingConfig, raw["training"], "training"),
            source=source,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f), source=str(path))

    def to_dict(self) -> dict[str, Any]:
        out = {
            "dataset": dataclasses.asdict(self.dataset),
            "model": dataclasses.asdict(self.model),
            "training": dataclasses.asdict(self.training),
        }
        out["training"]["betas"] = list(self.training.betas)
        return out


def _build(cls: type, values: dict[str, Any], section: str) -> Any:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys in '{section}' section: {sorted(unknown)}")
    return cls(**values)


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into ``{"a.b": value}`` form (used for experiment tracking)."""
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out
