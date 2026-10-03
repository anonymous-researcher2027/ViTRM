"""ViTRM: a Vision Transformer with TRM-style recursive reasoning.

The model keeps three token streams:

* ``x``: patch embeddings of the input image, shape ``(B, L, D)``;
* ``y``: a single global answer token, shape ``(B, 1, D)``;
* ``z``: a per-patch latent reasoning state, shape ``(B, L, D)``.

One recursion step (:meth:`ViTRM.deep_recursion`) updates the latent state
``n_latent_steps`` times with ``z <- f([x; y; z])`` and then the answer with
``y <- f([y; z])``. Both updates use the same three-layer transformer block ``f``. The
class logits and a halting logit are read from ``y``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vitrm.config import ViTRMConfig

NUM_LAYERS = 3  # transformer layers in the shared block


class PatchEmbed(nn.Module):
    """Non-overlapping patch embedding (strided convolution) with learned positions."""

    def __init__(self, image_size: int, patch_size: int, in_chans: int, dim: int) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.xavier_normal_(self.pos_embed)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size} images")
        tokens = self.proj(images).flatten(2).transpose(1, 2)  # (B, L, D)
        return tokens + self.pos_embed


class SwiGLU(nn.Module):
    """Split the last dimension in two halves ``(x, gate)`` and return ``x * silu(gate)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer: self-attention followed by a SwiGLU MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * mlp_hidden_dim),
            SwiGLU(),
            nn.Linear(mlp_hidden_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout(self.attn(h, h, h, need_weights=False)[0])
        return x + self.dropout(self.mlp(self.norm2(x)))


class RecursionOutput(NamedTuple):
    y: torch.Tensor
    z: torch.Tensor
    logits: torch.Tensor
    halt_logit: torch.Tensor


class ViTRM(nn.Module):
    def __init__(self, cfg: ViTRMConfig, num_classes: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.in_chans, cfg.dim)
        num_patches = self.patch_embed.num_patches

        self.y_init = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.z_init = nn.Parameter(torch.zeros(1, num_patches, cfg.dim))

        # A single block shared by the latent (z) and answer (y) updates.
        self.block = nn.Sequential(
            *[
                TransformerLayer(cfg.dim, cfg.num_heads, cfg.mlp_hidden_dim, cfg.dropout)
                for _ in range(NUM_LAYERS)
            ]
        )

        self.cls_head = nn.Linear(cfg.dim, num_classes)
        self.halt_head = nn.Linear(cfg.dim, 1)

        for param in (self.y_init, self.z_init, self.cls_head.weight, self.halt_head.weight):
            nn.init.xavier_normal_(param)
        nn.init.zeros_(self.cls_head.bias)
        nn.init.zeros_(self.halt_head.bias)

    def init_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.y_init.expand(batch_size, -1, -1)
        z = self.z_init.expand(batch_size, -1, -1)
        return y, z

    def latent_step(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``z <- f([x; y; z])``, keeping only the z slice of the output."""
        out = self.block(torch.cat([x, y, z], dim=1))
        return out[:, x.size(1) + y.size(1) :]

    def answer_step(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``y <- f([y; z])``, keeping only the y slice of the output."""
        out = self.block(torch.cat([y, z], dim=1))
        return out[:, : y.size(1)]

    def deep_recursion(
        self,
        images: torch.Tensor,
        y: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> RecursionOutput:
        """Run one recursion step. Pass ``y = z = None`` to start from the learned initial state."""
        x = self.patch_embed(images)
        if self.training and self.cfg.token_dropout > 0:
            x = F.dropout(x, p=self.cfg.token_dropout)
        if y is None or z is None:
            y, z = self.init_state(images.size(0))

        for _ in range(self.cfg.n_latent_steps):
            if self.training and self.cfg.latent_dropout > 0:
                z = F.dropout(z, p=self.cfg.latent_dropout)
            z = self.latent_step(x, y, z)
        y = self.answer_step(y, z)

        answer = y[:, 0]
        return RecursionOutput(y, z, self.cls_head(answer), self.halt_head(answer).squeeze(-1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Inference: chain ``t_deep`` recursion steps and return the final class logits."""
        y = z = None
        for _ in range(self.cfg.t_deep):
            y, z, logits, _ = self.deep_recursion(images, y, z)
        return logits
