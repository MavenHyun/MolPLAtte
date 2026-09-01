"""Assembly loss — cross-entropy over recovered joint chemistry.

The recovery branch is a set of small classifiers, one per attribute, so the loss
is a mean of cross-entropies.  MolDAM originally *summed* ~16 of them, which let
recovery outweigh its coupling term by roughly 13x and starve it of gradient; it
now divides by the contributing head count and so does this.  Averaging also
keeps the assembly term on a comparable scale to the three contrastive losses,
which matters because MolPLA's own stated limitation is "adversarial optimization
trajectories incurred by three different loss objectives" -- adding a fourth on an
uncontrolled scale would make that worse.

Per-attribute accuracies are surfaced as eval metrics rather than folded into one
number: recovering ``atomic_num`` at 99% while ``bond_type`` sits at 60% is a very
different situation from the reverse, and a macro average hides it.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["AssemblyLoss"]


class AssemblyLoss(nn.Module):
    """Recovery cross-entropy, plus optional joint coupling InfoNCE.

    Parameters
    ----------
    recovery_weight, coupling_weight
        Relative weights of the two branches.  ``coupling_weight`` is only used
        if the head was built with ``coupling=True``.
    temperature
        For the coupling InfoNCE.
    """

    def __init__(
        self,
        recovery_weight: float = 1.0,
        coupling_weight: float = 0.0,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.recovery_weight = recovery_weight
        self.coupling_weight = coupling_weight
        self.temperature = temperature

    def forward(self, batch: Dict) -> torch.Tensor:
        device = batch["W"].node_embeddings.device
        total = torch.zeros((), device=device)

        atom_pred: Dict = batch.get("assembly_atom_pred") or {}
        bond_pred: Dict = batch.get("assembly_bond_pred") or {}
        atom_true: Dict = batch.get("joint_atom_target") or {}
        bond_true: Dict = batch.get("joint_bond_target") or {}

        losses = []
        for pred, true in ((atom_pred, atom_true), (bond_pred, bond_true)):
            for attr, logits in pred.items():
                target = true.get(attr)
                if target is None or logits.numel() == 0:
                    continue
                if target.shape[0] != logits.shape[0]:
                    # A target/prediction length mismatch means the collate
                    # dropped joints the head still predicted for. Skip rather
                    # than align by truncation, which would pair wrong rows.
                    continue
                losses.append(F.cross_entropy(logits, target.to(logits.device)))
        if losses:
            # Mean, not sum: see the module docstring.
            total = total + self.recovery_weight * torch.stack(losses).mean()
            batch["loss/assembly_recovery"] = torch.stack(losses).mean().detach()

        if self.coupling_weight > 0:
            ce = batch.get("assembly_core_embedding")
            re_ = batch.get("assembly_rgroup_embedding")
            if ce is not None and re_ is not None and ce.shape[0] >= 2:
                logits = (ce @ re_.t()) / self.temperature
                labels = torch.arange(ce.shape[0], device=logits.device)
                coupling = 0.5 * (
                    F.cross_entropy(logits, labels)
                    + F.cross_entropy(logits.t(), labels)
                )
                total = total + self.coupling_weight * coupling
                batch["loss/assembly_coupling"] = coupling.detach()

        return total

    @torch.no_grad()
    def eval_metrics(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Per-attribute recovery accuracy, each beside its majority-class rate.

        Raw accuracy on these targets is close to meaningless on its own: some are
        near-constant, and two are *structurally* constant on a washed,
        non-ring-cut corpus --

        ``edge_is_aromatic`` a reformed cut bond is never aromatic. Measured on
                            coconut-flavordb_v4: 1 class, 100%. Still holds even
                            though ``_filter_cleavable`` now permits ring bonds
                            outside small rings -- a macrocycle bond is cuttable
                            but not aromatic.
        ``formal_charge``   NO LONGER degenerate. This previously read "wash()
                            neutralises charges, so a joint atom is always
                            neutral; measured 1 class". MolPLAtte passes
                            ``neutralise=False`` (organic acids and quaternary
                            ammonium tastants are chemically load-bearing in
                            flavour), so it now measures 2 classes -- but at
                            99.9% majority. That is worse than a clean
                            degenerate: the rare class appears in only some
                            batches, so the target flickers in and out of the
                            macro average batch to batch. Read its ``headroom``,
                            not the macro mean, when judging it.

        Both score ~100% for free and inflate the macro average. So every
        attribute also reports ``majority`` (the constant-predictor rate on this
        batch) and ``headroom`` (the fraction of the gap above it that the model
        actually closed), and the macro average is taken over **non-degenerate**
        targets only. Same discipline as ``library/lift@K``: an accuracy without
        its baseline is not a result.
        """
        out: Dict[str, torch.Tensor] = {}
        accs = []
        for pred_key, true_key, tag in (
            ("assembly_atom_pred", "joint_atom_target", "atom"),
            ("assembly_bond_pred", "joint_bond_target", "bond"),
        ):
            pred = batch.get(pred_key) or {}
            true = batch.get(true_key) or {}
            for attr, logits in pred.items():
                target = true.get(attr)
                if target is None or logits.numel() == 0:
                    continue
                if target.shape[0] != logits.shape[0]:
                    continue
                t = target.to(logits.device)
                acc = (logits.argmax(dim=-1) == t).float().mean()
                out[f"assembly/{tag}/{attr}"] = acc

                # Constant-predictor rate on this batch.
                counts = torch.bincount(t)
                majority = counts.max().float() / max(t.numel(), 1)
                out[f"assembly/{tag}/{attr}/majority"] = majority
                gap = 1.0 - majority
                if gap > 1e-6:
                    out[f"assembly/{tag}/{attr}/headroom"] = (acc - majority) / gap
                    accs.append(acc)
                # else: degenerate on this corpus -- excluded from the macro mean
                # rather than allowed to pad it toward 1.0.
        if accs:
            out["assembly/recover_acc"] = torch.stack(accs).mean()
        return out
