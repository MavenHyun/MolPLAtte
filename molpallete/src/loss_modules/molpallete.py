"""MolPallete's three-objective loss module.

Total loss is the weighted sum of the three contrastive terms::

    L = w_graph * L_graph + w_linker * L_linker + w_rgroup * L_rgroup

The paper reports all three weights as 1 (Eq. 14); the released implementation
uses 1.0 / 0.1 / 1.0.  MolPallete follows the implementation and exposes the
weights on the LightningModule, because the paper's own limitations section
attributes MolPLA's non-synergistic results to "adversarial optimization
trajectories incurred by three different loss objectives" -- i.e. the balance is
an open question, not a settled default.

Evaluation metrics are reported for the R-group retrieval objective only, since
that is the one with a task interpretation (lead optimization).  R@K is
**collision-aware**: a retrieval counts as a hit if the retrieved row shares the
query's chemistry key, not merely its row index.  With an effective R-group
vocabulary in the low hundreds, index-matched R@K systematically undercounts.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .base import LossModule
from .contrastive import DualInfoNCE, group_ids_from_keys

__all__ = ["LossModuleMolPallete"]


class LossModuleMolPallete(LossModule):
    def __init__(
        self,
        model,
        loss_graph_kwargs: Optional[dict] = None,
        loss_linker_kwargs: Optional[dict] = None,
        loss_rgroup_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)
        self.loss = torch.nn.ModuleDict(
            {
                "graph_contrastive": DualInfoNCE(**(loss_graph_kwargs or {})),
                "linker_contrastive": DualInfoNCE(**(loss_linker_kwargs or {})),
                "rgroup_contrastive": DualInfoNCE(**(loss_rgroup_kwargs or {})),
            }
        )

    def _compute_losses(self, batch: Dict) -> Dict:
        keys = batch.get("R_hashes")
        batch["loss/graph_contrastive"] = self.loss["graph_contrastive"](
            *batch["graph_contrastive"]
        )
        # The linker objective is also per-detached-R-group, so it shares the
        # R-group chemistry keys -- two joints on identical R-groups are not
        # meaningful negatives for each other either.
        batch["loss/linker_contrastive"] = self.loss["linker_contrastive"](
            *batch["linker_contrastive"], keys
        )
        batch["loss/rgroup_contrastive"] = self.loss["rgroup_contrastive"](
            *batch["rgroup_contrastive"], keys
        )
        return batch

    @torch.no_grad()
    def compute_eval_metrics(self, batch: Dict) -> Dict[str, torch.Tensor]:
        z_C, z_R = batch.get("rgroup_contrastive", (None, None))
        if z_C is None or z_C.shape[0] < 2:
            return {}

        x = torch.nn.functional.normalize(z_C.float(), dim=-1)
        y = torch.nn.functional.normalize(z_R.float(), dim=-1)
        sim = x @ y.t()
        n = sim.shape[0]

        keys = batch.get("R_hashes")
        if keys and len(keys) == n:
            gid = group_ids_from_keys(keys, sim.device)
            same = gid.unsqueeze(1) == gid.unsqueeze(0)
        else:
            same = torch.eye(n, dtype=torch.bool, device=sim.device)

        metrics: Dict[str, torch.Tensor] = {}
        for k in (1, 5, 10):
            if k > n:
                continue
            topk = sim.topk(k, dim=1).indices
            metrics[f"retrieval/r@{k}"] = (
                same.gather(1, topk).any(dim=1).float().mean()
            )

        # MRR against the index-matched target, ties broken in its favour.
        target = sim.diagonal().unsqueeze(1)
        rank = (sim > target).sum(dim=1) + 1
        metrics["retrieval/mrr"] = (1.0 / rank.float()).mean()
        # How much of the batch is chemically duplicated -- the number that says
        # whether an R@K figure is measuring retrieval or measuring the prior.
        metrics["retrieval/dup_rate"] = (
            (same.sum(dim=1) > 1).float().mean()
        )
        metrics["retrieval/n_queries"] = torch.tensor(float(n), device=sim.device)
        return metrics
