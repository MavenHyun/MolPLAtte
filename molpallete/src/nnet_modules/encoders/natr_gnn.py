"""Node Attribute Transformer encoder (NATR) for MolPallete.

Implements Lee et al., "Understanding and Tackling Over-Dilution in Graph
Neural Networks" (KDD 2025), §4, as a drop-in replacement for ``VanillaGNN``.

Why this maps onto MolPallete cleanly
---------------------------------
NATR treats node attributes as tokens drawn from a global vocabulary T, where
node v holds a subset T_v. MolPallete already stores six categorical attributes per
atom, so the vocabulary is every (attribute-type, value) pair -- 174 tokens --
and every atom holds exactly six of them, one per type. No feature engineering
is needed; the tokens are already there.

Where the analogy strains
-------------------------
The paper's datasets have |T_v| ~ 200 with degree ~19, giving a per-attribute
influence near 0.5%. MolPallete has |T_v| = 6 and median degree 2, i.e. ~16.7%.
MolPallete also *concatenates* its attribute embeddings through a learned MLP
rather than summing them, so the uniform-1/|T_v| collapse the paper targets is
already partly avoided. Expect a smaller effect here than the paper reports.
`RepresentationHealth`'s `health/intra_dilution/entropy` measures exactly how
much room is left: 1.0 means the fusion collapsed to uniform after all.

Architecture (Fig. 3)
---------------------
    Attribute Encoder   N x [MHSA -> Add&Norm -> FFN -> Add&Norm] over the
                        174 global attribute tokens. Runs once per forward,
                        not once per node.

    Attribute Decoder   M x [ H = NodeModule(H, A)                  (MPNN)
                              O = MHA(Q=H, K=V=attr tokens of v)    (cross)
                              G = Norm((1-lam) H + lam O)           (Eq. 10)
                              H = Norm(FFN(G) + G) ]

Because each atom holds exactly six attributes, the decoder's masked attention
degenerates to a gather: K/V are six tokens per node rather than a 174-wide
masked softmax. That makes the cross-attention cheap.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import torch
import torch.nn as nn

from ..components import (
    FEAT2DIM_EDGE,
    FEAT2DIM_NODE,
    GraphSequential,
    _make_conv,
    _make_norm,
)

if TYPE_CHECKING:
    from data_modules.mol_features import MolPalleteData as MolPalleteDataBatch


class NatrGNN(nn.Module):
    """NATR encoder. Interface-compatible with :class:`VanillaGNN`."""

    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_dim   = kwargs["hidden_dim"]
        self.dropout_rate = kwargs["dropout_rate"]
        self.norm_method  = kwargs["norm_method"]
        self.gnn_conv     = kwargs["gnn_conv"]
        self.num_conv     = kwargs["num_conv"]

        # NATR-specific
        self.n_attr_encoder_layers = int(kwargs.get("natr_n_encoder_layers", 2))
        self.n_heads               = int(kwargs.get("natr_n_heads", 4))
        # lambda in Eq. 10: 0 => pure MPNN, 1 => pure attribute readout.
        self.lam                   = float(kwargs.get("natr_lambda", 0.5))
        # "sum" is faithful to the paper (h^(0) = sum_t z_t, the regime NATR's
        # decoder exists to repair). "mlp" keeps MolPallete's concat+MLP fusion and
        # adds NATR on top -- weaker test of the hypothesis, likely stronger
        # absolute numbers.
        self.initial_fusion        = str(kwargs.get("natr_initial_fusion", "sum"))
        assert self.initial_fusion in ("sum", "mlp")

        h = self.hidden_dim
        self.moduledict = nn.ModuleDict()

        # -- global attribute-token vocabulary ------------------------------
        # One token per (attribute type, value). Offsets let a node's six
        # categorical values index straight into the shared table.
        offsets, total = [], 0
        for _, nd in FEAT2DIM_NODE.items():
            offsets.append(total)
            total += nd + 1                       # +1 matches VanillaGNN padding
        self.register_buffer("attr_offsets",
                             torch.tensor(offsets, dtype=torch.long),
                             persistent=False)
        self.n_attr_tokens = total
        self.n_attr_types  = len(FEAT2DIM_NODE)
        self.moduledict["attribute_tokens"] = nn.Embedding(total, h)

        # -- attribute encoder: MHSA over the global token table ------------
        self.moduledict["attribute_encoder"] = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h, nhead=self.n_heads, dim_feedforward=2 * h,
                dropout=self.dropout_rate, activation="gelu",
                batch_first=True, norm_first=False),
            num_layers=self.n_attr_encoder_layers)

        # -- edge features: unchanged from VanillaGNN -----------------------
        for ef, ed in FEAT2DIM_EDGE.items():
            self.moduledict[f"embedding_{ef}"] = nn.Embedding(ed + 1, h)
        norm_pw = lambda d: _make_norm(self.norm_method, d, graph_aware=False)
        edge_in  = h * len(FEAT2DIM_EDGE)
        edge_mid = edge_in // 2
        self.moduledict["fusion_edge"] = nn.Sequential(
            nn.Linear(edge_in, edge_mid), norm_pw(edge_mid),
            nn.PReLU(), nn.Dropout(self.dropout_rate),
            nn.Linear(edge_mid, h), norm_pw(h),
            nn.PReLU(), nn.Dropout(self.dropout_rate),
        )

        # Optional MolPallete-style static fusion for initial_fusion="mlp".
        if self.initial_fusion == "mlp":
            node_in  = h * len(FEAT2DIM_NODE)
            node_mid = node_in // 2
            self.moduledict["fusion_node"] = nn.Sequential(
                nn.Linear(node_in, node_mid), norm_pw(node_mid),
                nn.PReLU(), nn.Dropout(self.dropout_rate),
                nn.Linear(node_mid, h), norm_pw(h),
                nn.PReLU(), nn.Dropout(self.dropout_rate),
            )

        # -- attribute decoder: M blocks, MPNN nested inside ----------------
        for m in range(self.num_conv):
            self.moduledict[f"node_module_{m}"] = GraphSequential(
                _make_conv(self.gnn_conv, in_dim=h, out_dim=h, edge_dim=h),
                _make_norm(self.norm_method, h, graph_aware=True),
                nn.PReLU(),
                nn.Dropout(self.dropout_rate),
            )
            self.moduledict[f"cross_attn_{m}"] = nn.MultiheadAttention(
                embed_dim=h, num_heads=self.n_heads,
                dropout=self.dropout_rate, batch_first=True)
            self.moduledict[f"norm_attn_{m}"] = nn.LayerNorm(h)
            self.moduledict[f"ffn_{m}"] = nn.Sequential(
                nn.Linear(h, 2 * h), nn.GELU(),
                nn.Dropout(self.dropout_rate), nn.Linear(2 * h, h))
            self.moduledict[f"norm_ffn_{m}"] = nn.LayerNorm(h)

        # Populated each forward: (N, n_attr_types) attention over a node's own
        # attribute tokens, averaged across decoder layers. This is NATR's
        # delta^intra (paper §6.1) read directly off the attention coefficients,
        # rather than via a Jacobian as MPNNs require.
        self.last_attribute_attention: torch.Tensor | None = None

    # ----------------------------------------------------------------------

    def _node_attribute_indices(self, batch) -> torch.Tensor:
        """(N, 6) global token index for each atom's six attribute values."""
        vals = [batch[nf].long() for nf in FEAT2DIM_NODE]
        return torch.stack(vals, dim=1) + self.attr_offsets

    def forward(self, batch: "MolPalleteDataBatch") -> "MolPalleteDataBatch":
        h = self.hidden_dim

        # -- attribute encoder (once per forward, over the whole vocabulary) --
        Z = self.moduledict["attribute_tokens"].weight.unsqueeze(0)   # (1, T, h)
        Z = self.moduledict["attribute_encoder"](Z).squeeze(0)        # (T, h)

        # -- per-node attribute tokens ---------------------------------------
        idx = self._node_attribute_indices(batch)                     # (N, 6)
        Zv  = Z[idx]                                                  # (N, 6, h)

        # -- initial node representation --------------------------------------
        if self.initial_fusion == "sum":
            # h^(0) = sum_t z_t -- the paper's formulation, and the one whose
            # 1/|T_v| dilution the decoder is meant to repair.
            H = Zv.sum(dim=1)
        else:
            H = self.moduledict["fusion_node"](Zv.reshape(Zv.size(0), -1))

        # -- edge features -----------------------------------------------------
        E = [self.moduledict[f"embedding_{ef}"](batch[ef].long())
             for ef in FEAT2DIM_EDGE]
        E = self.moduledict["fusion_edge"](torch.cat(E, dim=1))

        # -- attribute decoder --------------------------------------------------
        attn_acc: List[torch.Tensor] = []
        for m in range(self.num_conv):
            H = self.moduledict[f"node_module_{m}"](
                H, batch.edge_index, edge_attr=E, batch=batch.batch)

            # Cross-attend from the node representation to its own attributes.
            # Query is one token; keys/values are the node's six attributes, so
            # the paper's attention mask is implicit in the gather.
            O, w = self.moduledict[f"cross_attn_{m}"](
                H.unsqueeze(1), Zv, Zv, need_weights=True,
                average_attn_weights=True)
            O = O.squeeze(1)
            attn_acc.append(w.squeeze(1).detach())

            G = self.moduledict[f"norm_attn_{m}"]((1.0 - self.lam) * H
                                                  + self.lam * O)
            H = self.moduledict[f"norm_ffn_{m}"](
                self.moduledict[f"ffn_{m}"](G) + G)

        self.last_attribute_attention = (
            torch.stack(attn_acc, dim=0).mean(dim=0) if attn_acc else None)
        batch.node_embeddings = H
        return batch
