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

``basis_path`` addresses the capacity problem measured on 2026-09-09. The free
reduction is 1280 -> 128 -> 32, i.e. 168,096 parameters fitted from 243
tastepocket records -- 692 per record. A probe over the 1,255 pocket sites showed
the ESM-2 space DOES carry ligand chemistry across unseen receptors (Tanimoto
0.35 vs 0.12 for a random pocket), and that signal lives in the metric structure
of the space, which a reduction that underdetermined cannot preserve; the easiest
thing it can fit instead is receptor identity, which is useless on a held-out
receptor by construction.

With a basis, the 1280 -> 32 step becomes a FIXED orthonormal projection fitted
without labels (PCA over training-fold pockets only), and the only learned part
is a 32 -> 32 adapter: 1,056 parameters instead of 168,096. The metric structure
survives by construction because an orthonormal projection is close to an
isometry on the retained subspace.
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
        basis_path: Optional[str] = None,
        use_basis: bool = False,
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

        # Declared before any early return so `self.basis` always exists, and as
        # a buffer so `.to(device)` and state_dict round-trips cover it.
        self.register_buffer("basis", None)
        self.register_buffer("basis_mean", None)
        self.register_buffer("basis_scale", None)

        if self.pocket_input_dim == 0:
            self.project = None
            return

        # `use_basis` without a path builds the SAME shape with empty buffers,
        # to be filled by load_state_dict. The basis is saved in the checkpoint,
        # so reloading a basis-reduced model must not require the .npz that
        # produced it -- that file is a training artefact, not a runtime one.
        if basis_path or use_basis:
            if basis_path:
                self._load_basis(basis_path)
            else:
                self.basis = torch.zeros(self.pocket_dim, self.pocket_input_dim)
                self.basis_mean = torch.zeros(self.pocket_input_dim)
                self.basis_scale = torch.ones(self.pocket_dim)
            # Learned part is the adapter ALONE; the basis is a buffer, so it is
            # neither trained nor perturbed by a warm start.
            out = nn.Linear(self.pocket_dim, self.pocket_dim)
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
            self.project = nn.Sequential(nn.Dropout(dropout), out)
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

    def _load_basis(self, basis_path: str) -> None:
        """Install a fixed PCA projection as non-trainable buffers.

        The file must carry ``components`` [pocket_dim, pocket_input_dim] and
        ``mean`` [pocket_input_dim]. Shapes are checked rather than trusted: a
        basis fitted at a different width would broadcast into silence.
        """
        import numpy as np

        z = np.load(basis_path)
        for k in ("components", "mean"):
            if k not in z:
                raise KeyError(f"{basis_path} has no '{k}' array")
        comp = torch.as_tensor(z["components"], dtype=torch.float32)
        mean = torch.as_tensor(z["mean"], dtype=torch.float32)
        if comp.shape != (self.pocket_dim, self.pocket_input_dim):
            raise ValueError(
                f"basis components {tuple(comp.shape)} != expected "
                f"{(self.pocket_dim, self.pocket_input_dim)}"
            )
        if mean.shape != (self.pocket_input_dim,):
            raise ValueError(
                f"basis mean {tuple(mean.shape)} != expected "
                f"{(self.pocket_input_dim,)}"
            )
        scale = z["scale"] if "scale" in z else np.ones(self.pocket_dim)
        self.basis = comp
        self.basis_mean = mean
        self.basis_scale = torch.as_tensor(
            scale, dtype=torch.float32
        ).clamp_min(1e-6)

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

        if self.basis is not None:
            # Fixed orthonormal projection, then unit-variance per component so
            # the adapter and the 24 flavor bits start on comparable scales.
            reduced = (pocket - self.basis_mean) @ self.basis.t()
            reduced = reduced / self.basis_scale
            # NOT a residual: the adapter's zero-init must still mean "the
            # pocket contributes exactly nothing at step 0", or a warm start
            # would jump off the pretrained solution the moment it loads.
            # A zero-init 32x32 learns the pass-through it needs from 243
            # records; 168k free parameters could not.
            projected = self.project(reduced) * has_pocket
        else:
            projected = self.project(pocket) * has_pocket
        return torch.cat([flavor, projected], dim=-1)
