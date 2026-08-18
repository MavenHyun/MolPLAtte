"""Dual InfoNCE, plus the two refinements MolPLA's plain version needs here.

Base objective (paper Eq. 13), symmetric in both directions with cosine
similarity::

    L = -(1/2B) sum_i [ log( e^{sim(x_i,y_i)/T} / sum_j e^{sim(x_i,y_j)/T} )
                      + log( e^{sim(y_i,x_i)/T} / sum_j e^{sim(y_i,x_j)/T} ) ]

The released MolPLA code calls this ``dualentropy``; the paper calls it "dual
InfoNCE".  Both mean the same thing -- what is "dual" is the two directions, and
the score function is plain cosine similarity, not a different entropy.

Two additions, both aimed at the R-group retrieval objective specifically:

**Multi-positive masking.**  Plain InfoNCE treats every other row as a negative,
but R-group vocabularies are brutally skewed -- MolDAM measured a top-1 R-group
share of 27.9% with an *effective* vocabulary of ~120 distinct items.  At that
skew a large fraction of in-batch "negatives" are chemically identical to the
positive, so the objective has an irreducible floor and the gradient actively
pushes identical structures apart.  Rows sharing a WL subgraph hash are therefore
treated as mutual positives.

**logQ correction.**  Even with multi-positive masking, frequent R-groups
dominate the denominator.  Subtracting ``log q_k`` (a running frequency estimate
per chemistry key, add-one smoothed) turns the similarity into a likelihood ratio
against the frequency prior.  This matters because MolDAM's headline retrieval
number, re-scored against the frequency prior rather than against random, landed
*at* the prior rather than above it -- "better than random" is not a meaningful
claim when the prior is this strong.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["DualInfoNCE", "group_ids_from_keys"]


def group_ids_from_keys(keys: Sequence[str], device: torch.device) -> torch.Tensor:
    """Map chemistry keys to contiguous ids; equal keys share an id."""
    lookup: Dict[str, int] = {}
    ids: List[int] = []
    for key in keys:
        if key not in lookup:
            lookup[key] = len(lookup)
        ids.append(lookup[key])
    return torch.tensor(ids, dtype=torch.long, device=device)


class DualInfoNCE(nn.Module):
    """Symmetric InfoNCE with optional multi-positive masking and logQ correction.

    Parameters
    ----------
    temperature
        MolPLA's released settings use 0.1 / 0.05 / 0.01 for the graph, linker and
        R-group objectives; the paper reports 0.01 / 0.05 / 0.01.
    multi_positive
        Treat rows sharing a chemistry key as mutual positives.  Requires keys to
        be passed to :meth:`forward`; silently falls back to the identity mask
        when they are absent.
    logq_correction, logq_warmup_batches
        Subtract a running log-frequency estimate.  Until ``logq_warmup_batches``
        have been seen the batch-local estimate is used, which floors at
        ``1/batch_size`` and so understates how rare a rare key really is.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        multi_positive: bool = False,
        logq_correction: bool = False,
        logq_warmup_batches: int = 20,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature
        self.multi_positive = multi_positive
        self.logq_correction = logq_correction
        self.logq_warmup_batches = logq_warmup_batches
        self._counts: Counter = Counter()
        self._n_batches = 0

    def _positive_mask(
        self, n: int, keys: Optional[Sequence[str]], device: torch.device
    ) -> torch.Tensor:
        if not self.multi_positive or not keys or len(keys) != n:
            return torch.eye(n, device=device)
        gid = group_ids_from_keys(keys, device)
        return (gid.unsqueeze(1) == gid.unsqueeze(0)).float()

    def _log_q(
        self, keys: Optional[Sequence[str]], mask: torch.Tensor, device: torch.device
    ) -> Optional[torch.Tensor]:
        if not self.logq_correction:
            return None
        n = mask.shape[0]
        if keys and len(keys) == n:
            if self.training:
                self._counts.update(keys)
                self._n_batches += 1
            if self._n_batches >= self.logq_warmup_batches and self._counts:
                total = sum(self._counts.values())
                n_keys = len(self._counts)
                probs = [
                    (self._counts[k] + 1) / (total + n_keys) for k in keys
                ]
                return torch.log(torch.tensor(probs, device=device)).clamp(min=-30.0)
        # Batch-local fallback: how many rows share each row's chemistry.
        return torch.log((mask.sum(dim=0) / n).clamp(min=1e-12))

    def forward(
        self,
        view1: Optional[torch.Tensor],
        view2: Optional[torch.Tensor],
        keys: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        if view1 is None or view2 is None or view1.numel() == 0:
            return torch.zeros((), device=view1.device if view1 is not None else "cpu")
        n = view1.shape[0]
        if n < 2:
            # A single pair has no negatives; the softmax is degenerate at 1.
            return view1.new_zeros(())

        x = F.normalize(view1, dim=-1)
        y = F.normalize(view2, dim=-1)
        logits = (x @ y.t()) / self.temperature

        mask = self._positive_mask(n, keys, logits.device)
        log_q = self._log_q(keys, mask, logits.device)

        def direction(lg: torch.Tensor, mk: torch.Tensor) -> torch.Tensor:
            if log_q is not None:
                lg = lg - log_q.unsqueeze(0)
            log_p = F.log_softmax(lg, dim=1)
            n_pos = mk.sum(dim=1).clamp(min=1.0)
            return (-(log_p * mk).sum(dim=1) / n_pos).mean()

        return 0.5 * direction(logits, mask) + 0.5 * direction(logits.t(), mask.t())
