from __future__ import annotations

import pytest
import torch

from conftest import tiny_config
from vitrm.models import ViTRM, build_model


def test_vitrm_shapes():
    model = ViTRM(tiny_config().model, num_classes=10).eval()
    images = torch.randn(4, 3, 32, 32)

    y, z, logits, halt_logit = model.deep_recursion(images)
    assert y.shape == (4, 1, 32)
    assert z.shape == (4, 64, 32)  # (32 / 4) ** 2 patches
    assert logits.shape == (4, 10)
    assert halt_logit.shape == (4,)
    assert model(images).shape == (4, 10)


def test_vitrm_forward_chains_t_deep_recursions():
    model = ViTRM(tiny_config(t_deep=3).model, num_classes=10).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = z = None
        for _ in range(3):
            y, z, logits, _ = model.deep_recursion(images, y, z)
        torch.testing.assert_close(model(images), logits)


def test_vitrm_eval_is_deterministic_and_train_mode_uses_dropout():
    model = ViTRM(tiny_config(token_dropout=0.5).model, num_classes=10)
    images = torch.randn(2, 3, 32, 32)
    model.eval()
    torch.testing.assert_close(model(images), model(images))
    model.train()
    assert not torch.allclose(model(images), model(images))


def test_vitrm_shared_block_has_three_layers():
    assert len(ViTRM(tiny_config().model, num_classes=10).block) == 3


@pytest.mark.parametrize("family", ["vitrm", "vit", "resnet"])
def test_build_model(family):
    model = build_model(tiny_config(family).model, num_classes=100).eval()
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 100)
