from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..components import (
    EDGE_BLIND_CONVS,
    FEAT2DIM_EDGE,
    FEAT2DIM_NODE,
    GraphSequential,
    _make_conv,
    _make_norm,
)

if TYPE_CHECKING:
    from data_modules.mol_features import MolPLAtteData as MolPLAtteDataBatch


class VanillaGNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_dim   = kwargs['hidden_dim']
        self.dropout_rate = kwargs['dropout_rate']
        self.norm_method  = kwargs['norm_method']
        self.gnn_conv     = kwargs['gnn_conv']
        self.num_conv     = kwargs['num_conv']
        self.moduledict   = nn.ModuleDict()

        for nf, nd in FEAT2DIM_NODE.items():
            self.moduledict[f'embedding_{nf}'] = nn.Embedding(nd+1, self.hidden_dim)
        for ef, ed in FEAT2DIM_EDGE.items():
            self.moduledict[f'embedding_{ef}'] = nn.Embedding(ed+1, self.hidden_dim)

        # Per-node / per-edge fusion MLPs run before message passing —
        # no ``batch`` index is threaded through ``nn.Sequential``, so
        # use the pointwise norm fallback (graph_aware=False).
        node_in,  node_mid  = self.hidden_dim*len(FEAT2DIM_NODE), self.hidden_dim*len(FEAT2DIM_NODE)//2
        edge_in,  edge_mid  = self.hidden_dim*len(FEAT2DIM_EDGE), self.hidden_dim*len(FEAT2DIM_EDGE)//2
        norm_pw = lambda d: _make_norm(self.norm_method, d, graph_aware=False)

        self.moduledict['fusion_node'] = nn.Sequential(
            nn.Linear(node_in, node_mid),
            norm_pw(node_mid),
            nn.PReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(node_mid, self.hidden_dim),
            norm_pw(self.hidden_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.moduledict['fusion_edge'] = nn.Sequential(
            nn.Linear(edge_in, edge_mid),
            norm_pw(edge_mid),
            nn.PReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(edge_mid, self.hidden_dim),
            norm_pw(self.hidden_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout_rate),
        )

        conv_blocks: list[nn.Module] = []
        for _ in range(self.num_conv):
            conv_blocks.append(_make_conv(self.gnn_conv,
                                          in_dim=self.hidden_dim,
                                          out_dim=self.hidden_dim,
                                          edge_dim=self.hidden_dim))
            conv_blocks.append(_make_norm(self.norm_method, self.hidden_dim, graph_aware=True))
            conv_blocks.append(nn.PReLU())
            conv_blocks.append(nn.Dropout(self.dropout_rate))
        self.moduledict['graph_conv'] = GraphSequential(*conv_blocks)


    def forward(self, batch: "MolPLAtteDataBatch") -> "MolPLAtteDataBatch":
        # Per-atom / per-bond attributes are stored as int8 on disk to
        # cut LMDB size 6×; nn.Embedding needs long indices, so upcast
        # once per attribute right at the lookup site.
        X = [self.moduledict[f'embedding_{nf}'](batch[nf].long()) for nf in FEAT2DIM_NODE.keys()]
        X = self.moduledict['fusion_node'](torch.cat(X, dim=1))

        E = [self.moduledict[f'embedding_{ef}'](batch[ef].long()) for ef in FEAT2DIM_EDGE.keys()]
        E = self.moduledict['fusion_edge'](torch.cat(E, dim=1))

        batch.node_embeddings = self.moduledict['graph_conv'](
            X, batch.edge_index, edge_attr=E, batch=batch.batch,
        )
        return batch

