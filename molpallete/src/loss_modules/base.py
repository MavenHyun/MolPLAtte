"""Loss-module contract.

Composition is ``LightningModule(LossModule(Model))``: the loss module owns the
model, runs it, then computes losses onto the same batch dict.  Subclasses
populate ``batch["loss/<name>"]`` and may override
:meth:`compute_eval_metrics` to return a flat ``{name: scalar}`` dict that the
LightningModule logs as ``{stage}/<name>``.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

__all__ = ["LossModule"]


class LossModule(nn.Module):
    def __init__(self, model: nn.Module, **kwargs: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: Dict) -> Dict:
        batch = self.model(batch)
        batch = self._compute_losses(batch)
        return batch

    def _compute_losses(self, batch: Dict) -> Dict:
        raise NotImplementedError

    def compute_eval_metrics(self, batch: Dict) -> Dict[str, torch.Tensor]:
        return {}
