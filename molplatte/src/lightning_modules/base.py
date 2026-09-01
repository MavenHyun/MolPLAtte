from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn
import pytorch_lightning as pl


class MolPLAtteLightningModule(pl.LightningModule):
    """Lightning wrapper around a plain ``nn.Module`` MolPLAtte model.

    The wrapped model's ``forward(batch: dict) -> dict`` is expected to
    populate ``batch["loss/<name>"]`` entries (one scalar per loss
    component). This module then:
      * weighted-sums them into the total training/validation loss
      * logs every per-component loss + the total under
        ``{stage}/loss/...`` for wandb / tensorboard
      * builds an Adam optimizer from ``learning_rate`` / ``weight_decay``

    The three components are MolPLA's three contrastive objectives:

      ``graph_contrastive``   pooled ``G``  <-> pooled ``Q``   (B = #instances)
      ``linker_contrastive``  linker node emb in ``G`` <-> ``P_lj + R_lj``
                                                            (B = #linker nodes)
      ``rgroup_contrastive``  ``[P linker node emb || condvec]`` <-> pooled ``R``
                                                            (B = #(query, R) pairs)

    Each has its own effective batch size, so the ``batch_size=`` passed to
    ``self.log`` is deliberately the *instance* count -- it is only used to
    weight epoch-level averages, and the instance count is the one quantity
    every component is commensurate with.

    The DataLoader emits a ``dict`` containing PyG ``Batch`` objects,
    plain tensors, nested dicts of tensors, and string lists; the
    overridden :py:meth:`transfer_batch_to_device` moves anything with
    ``.to(...)`` and leaves the rest alone.
    """

    #: MolPLA's released ``settings.yaml`` coefficients. The paper (btae256,
    #: Eq. 14) reports all three at 1.0; the released code downweights the
    #: linker term to 0.1. We follow the code, because those are the weights
    #: the published numbers were actually produced with.
    DEFAULT_LOSS_WEIGHTS: Dict[str, float] = {
        "graph_contrastive":  1.0,
        "linker_contrastive": 0.1,
        "rgroup_contrastive": 1.0,
    }

    def __init__(self, model: nn.Module, **kwargs: Any):
        super().__init__()
        self.model            = model
        self.learning_rate    = kwargs.get("learning_rate", 1.0e-3)
        self.weight_decay     = kwargs.get("weight_decay",  0.0)
        self.use_cosine_lr    = bool(kwargs.get("use_cosine_lr", False))
        self.cosine_t_max     = int(kwargs.get("cosine_t_max", 100))
        self.cosine_eta_min   = float(kwargs.get("cosine_eta_min", 1.0e-6))
        self.loss_weights:    Dict[str, float] = dict(
            kwargs.get("loss_weights") or self.DEFAULT_LOSS_WEIGHTS)
        self.loss_components: List[str]        = list(self.loss_weights) or [
            "graph_contrastive", "linker_contrastive", "rgroup_contrastive"]
        self.sub_components:  List[str] = list(kwargs.get("sub_components") or [])

    def forward(self, batch: Dict) -> Dict:
        return self.model(batch)

    def _shared_step(self, batch: Dict, stage: str) -> torch.Tensor:
        batch = self.model(batch)
        B     = batch["G"].num_graphs

        total = torch.zeros((), device=self.device)
        for name in self.loss_components:
            key = f"loss/{name}"
            if key not in batch:
                continue
            w = self.loss_weights.get(name, 1.0)
            self.log(f"{stage}/{key}", batch[key],
                     on_step=(stage == "train"), on_epoch=True,
                     prog_bar=True, batch_size=B)
            total = total + w * batch[key]

        for sub in self.sub_components:
            key = f"loss/{sub}"
            if key in batch:
                self.log(f"{stage}/{key}", batch[key],
                         on_step=False, on_epoch=True, batch_size=B)

        if hasattr(self.model, "compute_eval_metrics"):
            for k, v in self.model.compute_eval_metrics(batch).items():
                self.log(f"{stage}/{k}", v,
                         on_step=False, on_epoch=True, batch_size=B)

        self.log(f"{stage}/loss", total,
                 on_step=(stage == "train"), on_epoch=True,
                 prog_bar=True, batch_size=B)
        return total

    def training_step(self,   batch, batch_idx): return self._shared_step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._shared_step(batch, "val")
    def test_step(self,       batch, batch_idx): return self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        if not self.use_cosine_lr:
            return opt
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cosine_t_max, eta_min=self.cosine_eta_min)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int = 0):
        def _move(v):
            if hasattr(v, "to"):
                return v.to(device, non_blocking=True)
            if isinstance(v, dict):
                return {k: _move(x) for k, x in v.items()}
            return v
        return {k: _move(v) for k, v in batch.items()}
