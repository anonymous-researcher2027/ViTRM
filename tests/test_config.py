from __future__ import annotations

import pytest

from conftest import CONFIG_DIR, tiny_config
from vitrm.config import ExperimentConfig, ResNetConfig, ViTConfig, ViTRMConfig

ALL_CONFIGS = sorted(CONFIG_DIR.rglob("*.yaml"))
FAMILY_BY_DIR = {"vitrm": ViTRMConfig, "vit": ViTConfig, "resnet": ResNetConfig}


def test_configs_exist():
    assert len(ALL_CONFIGS) > 100


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: str(p.relative_to(CONFIG_DIR)))
def test_shipped_config_is_valid(path):
    config = ExperimentConfig.from_yaml(path)
    model_dir, dataset_dir = path.relative_to(CONFIG_DIR).parts[:2]
    assert config.dataset.name == dataset_dir
    assert isinstance(config.model, FAMILY_BY_DIR[model_dir.split("-")[0].rstrip("0123456789")])
    if path.stem.startswith("batch_"):
        assert config.training.batch_size == int(path.stem.split("_")[1])


def test_roundtrip_through_dict():
    config = tiny_config()
    assert ExperimentConfig.from_dict(config.to_dict()) == config


def test_unknown_key_is_rejected():
    raw = tiny_config().to_dict()
    raw["model"]["n_latent_step"] = 3  # typo
    with pytest.raises(ValueError, match="Unknown keys in 'model'"):
        ExperimentConfig.from_dict(raw)


def test_unknown_family_is_rejected():
    raw = tiny_config().to_dict()
    raw["model"]["family"] = "mlp"
    with pytest.raises(ValueError, match="model.family"):
        ExperimentConfig.from_dict(raw)


def test_string_learning_rate_is_coerced():
    raw = tiny_config().to_dict()
    raw["training"]["learning_rate"] = "3e-4"
    assert ExperimentConfig.from_dict(raw).training.learning_rate == pytest.approx(3e-4)
