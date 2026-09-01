"""Projection heads for MolPLA's three objectives.

MolPLA names four heads (paper Fig. 2): ``g_theta`` graph projection, ``g_kappa``
node projection, ``g_Phi`` query linker-node projection and ``g_phi`` R-group
graph projection.  All are MLPs with LeakyReLU; the paper does not state their
depth or width, so :class:`MLPProjector` follows the released implementation's
two-layer shape.

Two conventions worth stating because MolDAM diverges from both:

*Sharing.*  ``g_theta`` and ``g_kappa`` are each applied to **both** sides of
their objective -- G and Q share the graph projector, G and (P+R) share the node
projector.  MolDAM instead gives each side its own projection.  MolPLAtte shares,
because an asymmetric projection lets the two branches drift into separate
subspaces and the contrastive objective can be satisfied without the encoder
learning anything shared.

*Conditioning.*  ``g_Phi`` alone takes a wider input: the core-side linker node
embedding concatenated with the R-group condition vector (paper Eq. 11), so its
input dimension is ``hidden_dim + condvec_dim``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..components.utils import _make_norm

__all__ = ["MLPProjector"]


class MLPProjector(nn.Module):
    """``Linear -> Norm -> LeakyReLU -> Dropout -> Linear``.

    Parameters
    ----------
    input_dim
        Defaults to ``hidden_dim``; set it wider for the conditioned query head.
    output_dim
        Defaults to ``hidden_dim``.
    norm_method
        Any norm name resolvable by ``components.utils._make_norm``.  Resolved
        with ``graph_aware=False`` because projections act on flat node/graph
        vectors that carry no batch vector.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        hidden_dim: int = kwargs["hidden_dim"]
        input_dim: int = kwargs.get("input_dim") or hidden_dim
        output_dim: int = kwargs.get("output_dim") or hidden_dim
        dropout_rate: float = kwargs.get("dropout_rate", 0.0)
        norm_method: Optional[str] = kwargs.get("norm_method", "LayerNorm")

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _make_norm(norm_method, hidden_dim, graph_aware=False),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
