from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    """Exponential moving average of model weights.

    ``EMA.module`` is a frozen copy of the model holding the averaged parameters; buffers
    (e.g. BatchNorm statistics) are copied from the live model at every update.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_param, param in zip(self.module.parameters(), model.parameters()):
            ema_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        for ema_buf, buf in zip(self.module.buffers(), model.buffers()):
            ema_buf.copy_(buf)
