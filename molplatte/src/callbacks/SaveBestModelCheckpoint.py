from __future__ import annotations

import os
import logging
from typing import Optional

import torch

from checkpoint_spec import write_sidecar
import pytorch_lightning as pl


class SaveBestModelCheckpoint(pl.Callback):
    """Save the **underlying nnet model** (not the loss/Lightning wrappers)
    whenever a watched val metric improves.

    Default watches ``val/loss`` written by
    :class:`lightning_modules.MolPLAtteLightningModule`. Saves
    ``model.state_dict()`` where ``model`` is two levels in:
    ``LightningModule.model``      →  ``LossModuleMolPLAtte``
    ``LossModuleMolPLAtte.model`` →  ``MolPLAtte``  ← this gets saved.

    Always saves only on global_rank=0 (DDP-safe).
    """

    def __init__(self,
                 save_dir: str,
                 monitor:  str  = "val/loss",
                 mode:     str  = "min",
                 filename: str  = "best_model.pt"):
        super().__init__()
        self.save_path = os.path.join(save_dir, filename)
        self.monitor   = monitor
        self.mode      = mode
        self.best      = float("inf") if mode == "min" else float("-inf")

    def _is_better(self, current: float) -> bool:
        return (current < self.best) if self.mode == "min" else (current > self.best)

    def on_train_epoch_end(self, trainer, pl_module):
        """Also save when there is no validation loop to hang off.

        The final deliverable checkpoint trains on ALL records with
        limit_val_batches=0, so on_validation_epoch_end never fires. Without
        this the run completes normally, logs a full set of training metrics,
        and writes no checkpoint at all -- a silent failure that only shows up
        when someone goes looking for the file.

        Guarded on the monitor being a TRAIN metric, not merely on it being
        present. callback_metrics persists across epochs, so from epoch 2 on a
        normal run would re-enter here with the PREVIOUS epoch's val/loss and
        compare a stale value.
        """
        if self.monitor.startswith("train"):
            self._save_if_better(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._save_if_better(trainer, pl_module)

    def _save_if_better(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        current = float(current)
        if not self._is_better(current):
            return
        self.best = current
        if trainer.global_rank != 0:
            return
        # LightningModule.model = LossModuleMolPLAtte; .model.model = nnet
        target = pl_module.model.model if hasattr(pl_module.model, "model") else pl_module.model
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(target.state_dict(), self.save_path)
        # A sidecar recording how to rebuild this. These are plain state_dicts
        # with no config in them, so without it every consumer has to restate
        # the architecture by hand -- and a wrong restatement loads a model that
        # runs and is wrong. See checkpoint_spec.
        try:
            cfg = getattr(target, "config", None)
            if cfg is not None:
                kw = {"condvec_dim": int(getattr(cfg, "condvec_dim", 0)),
                      "pocket_input_dim": int(getattr(cfg, "pocket_input_dim", 0)),
                      "pocket_dim": int(getattr(cfg, "pocket_dim", 0))}
                head = getattr(cfg, "assembly_head", None)
                if head:
                    kw["assembly_head"] = head
                # use_basis belongs IN model_kwargs -- it is needed to BUILD
                # the module. The path is recorded alongside for provenance
                # only; the basis itself is in the checkpoint.
                basis = getattr(cfg, "pocket_basis_path", None)
                if basis:
                    kw["use_basis"] = True
                write_sidecar(self.save_path, kw,
                              monitor=self.monitor, value=current,
                              pocket_basis_path=basis)
        except Exception as exc:  # noqa: BLE001 - never lose a checkpoint over metadata
            logging.getLogger(__name__).warning(
                "could not write checkpoint sidecar: %s", exc)
        logging.getLogger(__name__).info(
            f"Saved best model ({self.monitor}={current:.4f}) → {self.save_path}")
