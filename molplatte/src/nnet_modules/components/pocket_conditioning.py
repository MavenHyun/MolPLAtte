"""Pocket half of the condition vector: raw protein embedding -> query features.

The corpus stores a ``TwoPartCondVec`` laid out as ``[flavor | pocket]``, and
``MolPLAtte`` concatenates it straight onto the linker node embedding before the
query projector. That plumbing is why the pocket half cannot be stored in the
form the query should see:

    hidden_dim 300 + flavor 24 + ESM-2 1280  ->  the pocket is 80% of the input

Twenty-four sparse 0/1 bits next to 1280 dense floats of much larger magnitude
is not a condition vector with two halves, it is a pocket vector with some noise
attached. So the corpus stores the raw embedding and this module reduces it,
inside the model, where the reduction is learned rather than guessed.

Four choices here are load-bearing:

* **The flavor bits pass through untouched.** Projecting all 1304 dimensions
  jointly would dilute 24 sparse bits into a dense mixture, and the flavor half
  is the part that has to survive STEP 1 pretraining intact.
* **The pocket half is normalised before projection.** ESM-2 activations have
  neither zero mean nor unit scale; concatenated raw, they would dominate the
  query projector's gradient by magnitude alone, independent of whether they
  carry signal.
* **An all-zero pocket half stays exactly zero.** STEP 1 trains ligand-only with
  the pocket zeroed, and LayerNorm's affine bias would otherwise turn "no
  pocket" into a learned constant, making flavor-only runs silently
  non-equivalent to ``condvec_dim=24``.
* **The output layer is zero-initialised.** STEP 1.5 then starts from exactly
  the STEP 1 solution and has to earn any pocket contribution, instead of
  beginning with a random perturbation of a converged model.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

__all__ = ["PocketConditioning"]


class PocketConditioning(nn.Module):
    """Reduce ``[flavor | raw pocket]`` to ``[flavor | projected pocket]``.

    Parameters
    ----------
    flavor_dim:
        Width of the flavor half, passed through unchanged. 24 for the current
        wire format; 0 is legal and gives a pocket-only condition vector.
    pocket_input_dim:
        Width of the stored pocket embedding, e.g. 1280 for ESM-2 650M. ``0``
        disables the pocket path entirely and makes this module an identity, so
        flavor-only runs can keep it in the graph without behaviour changing.
    pocket_dim:
        Width the pocket half is projected to. Keep it the same order as
        ``flavor_dim``; the point of the projection is that neither half of the
        condition vector drowns the other.
    hidden_dim:
        Width of the intermediate layer. Defaults to ``4 * pocket_dim``.
    dropout:
        Applied to the projected pocket half. This is the main capacity control
        for STEP 2, which fine-tunes on a few hundred complexes.
    pocket_dropout:
        Probability of zeroing a row's pocket half outright during training.
        Distinct from ``dropout``: it drops the *whole* condition, not units
        within it, so the model must stay able to answer from flavor alone and
        cannot collapse onto the pocket. Zero at eval.
    """

    def __init__(
        self,
        flavor_dim: int = 24,
        pocket_input_dim: int = 0,
        pocket_dim: int = 32,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        pocket_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if flavor_dim < 0 or pocket_input_dim < 0 or pocket_dim < 0:
            raise ValueError("condition vector widths must be non-negative")
        if not 0.0 <= pocket_dropout < 1.0:
            raise ValueError(f"pocket_dropout must be in [0, 1), got {pocket_dropout}")

        self.flavor_dim = int(flavor_dim)
        self.pocket_input_dim = int(pocket_input_dim)
        self.pocket_dim = int(pocket_dim) if pocket_input_dim else 0
        self.pocket_dropout = float(pocket_dropout)

        if self.pocket_input_dim == 0:
            self.project = None
            return

        hidden = int(hidden_dim) if hidden_dim else max(4 * self.pocket_dim, 1)
        out = nn.Linear(hidden, self.pocket_dim)
        # Zero-init the output layer so a freshly attached pocket encoder
        # contributes exactly nothing on step 0. Fine-tuning then departs from
        # the pretrained solution rather than from a random perturbation of it.
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)

        self.project = nn.Sequential(
            nn.LayerNorm(self.pocket_input_dim),
            nn.Linear(self.pocket_input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            out,
        )

    # -- widths ------------------------------------------------------------
    @property
    def in_features(self) -> int:
        """Condition-vector width the CORPUS stores."""
        return self.flavor_dim + self.pocket_input_dim

    @property
    def out_features(self) -> int:
        """Condition-vector width the QUERY PROJECTOR sees."""
        return self.flavor_dim + self.pocket_dim

    @property
    def is_identity(self) -> bool:
        return self.project is None

    def extra_repr(self) -> str:
        return (
            f"flavor={self.flavor_dim}, pocket={self.pocket_input_dim}"
            f"->{self.pocket_dim}, in={self.in_features}, out={self.out_features}"
        )

    # -- forward -----------------------------------------------------------
    def forward(self, condvec: torch.Tensor) -> torch.Tensor:
        """``[N, in_features]`` -> ``[N, out_features]``."""
        if condvec.dim() != 2:
            raise ValueError(f"condvec must be 2-D [N, D], got {tuple(condvec.shape)}")
        if condvec.shape[-1] != self.in_features:
            raise ValueError(
                f"condvec width {condvec.shape[-1]} != expected {self.in_features} "
                f"(flavor {self.flavor_dim} + pocket {self.pocket_input_dim}); "
                "the corpus and the pocket encoder config disagree"
            )
        if self.project is None:
            return condvec

        flavor = condvec[:, : self.flavor_dim]
        pocket = condvec[:, self.flavor_dim :]

        # An all-zero row means "no pocket for this molecule" -- every STEP 1
        # record, and any complex whose pocket could not be embedded. Without
        # this mask LayerNorm's affine bias maps zeros to a learned constant,
        # and a flavor-only run silently stops matching condvec_dim=24.
        has_pocket = (pocket.abs().sum(dim=-1, keepdim=True) > 0).to(condvec.dtype)

        if self.training and self.pocket_dropout > 0.0:
            keep = (
                torch.rand(
                    pocket.shape[0], 1, device=pocket.device, dtype=condvec.dtype
                )
                >= self.pocket_dropout
            ).to(condvec.dtype)
            has_pocket = has_pocket * keep

        projected = self.project(pocket) * has_pocket
        return torch.cat([flavor, projected], dim=-1)
