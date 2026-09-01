"""MolPallete-specific message-passing layers.

PyG convs that are almost usable but cannot carry a vector-valued ``edge_attr``
get adapted here rather than being dropped from the backbone menu.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GatedGraphConv
from torch_geometric.typing import Adj, OptTensor


class EdgeGatedGraphConv(GatedGraphConv):
    r"""GGNN (Li et al., 2016) with vector-valued edge conditioning.

    PyG's :class:`GatedGraphConv` scales each message by a *scalar*
    ``edge_weight``:

    .. math::
        m_i = \sum_{j \in N(i)} e_{j,i} \cdot \Theta_l h_j

    MolPallete carries a ``hidden_dim`` bond embedding per edge (five attributes:
    bond_type, aromaticity, conjugation, direction, stereo), so a scalar
    throws almost all of it away. Instead gate the message elementwise:

    .. math::
        m_i = \sum_{j \in N(i)} \sigma(W_g e_{j,i}) \odot (\Theta_l h_j)

    The GRU node update -- the part that makes GGNN distinct from GINE, and
    the reason to want it here -- is inherited untouched.

    ``num_layers`` is the recurrence depth *within* one block. The MolPallete
    encoder stacks ``num_conv`` blocks, so the default of 1 gives one
    propagation step per block with independent weights, matching how every
    other conv in the stack behaves. That departs from canonical GGNN, which
    shares one GRU across all steps -- set ``num_layers > 1`` for the
    literal formulation.
    """

    def __init__(self, out_channels: int, num_layers: int = 1,
                 edge_dim: int | None = None, **kwargs):
        super().__init__(out_channels, num_layers, **kwargs)
        if edge_dim is None:
            raise ValueError("EdgeGatedGraphConv requires edge_dim; use plain "
                             "GatedGraphConv if you have no edge features.")
        self.edge_gate = nn.Linear(edge_dim, out_channels)
        # The inherited message_and_aggregate() fast path ignores edge gating
        # entirely. It only fires for SparseTensor adjacency, but leaving it
        # enabled would silently drop bond information -- disable it.
        self.fuse = False

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "edge_gate"):
            self.edge_gate.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        if x.size(-1) > self.out_channels:
            raise ValueError("number of input channels must not exceed "
                             "out_channels")
        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        gate = None if edge_attr is None else torch.sigmoid(self.edge_gate(edge_attr))

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])
            # PyG compiles propagate()'s signature from this annotation. The
            # inherited one declares edge_weight: OptTensor, so it must be
            # re-declared here or propagate() rejects edge_gate.
            # propagate_type: (x: Tensor, edge_gate: OptTensor)
            m = self.propagate(edge_index, x=m, edge_gate=gate)
            x = self.rnn(m, x)
        return x

    def message(self, x_j: Tensor, edge_gate: OptTensor) -> Tensor:
        return x_j if edge_gate is None else edge_gate * x_j

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.out_channels}, "
                f"num_layers={self.num_layers})")
