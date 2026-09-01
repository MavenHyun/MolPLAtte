"""MolPLAtte — MolPLA's masked graph contrastive learning on flavor chemistry.

One encoder pass, three objectives
----------------------------------
``G``, ``P`` and every ``R`` arrive collated into a single graph batch ``W``.  The
shared encoder runs **once** over ``W``; the three views are then recovered by
boolean node masks.  This is MolPLA's central efficiency trait and the reason the
framework trains three objectives for roughly the cost of one.

Reading the outputs (paper equation numbers in brackets):

``graph_contrastive`` [5-8]
    ``z_G = g_theta(pool(H[G]))`` against ``z_Q = g_theta(pool(H[P u R]))``.
    ``Q`` is the union of the query template and the detached R-groups, pooled as
    one graph per instance.

``linker_contrastive`` [9-10]
    ``z_m = g_kappa(m_i)`` against ``z_p = g_kappa(q_i + r_i)``, where ``m_i`` is
    the intact linker atom in ``G`` and ``q_i``/``r_i`` are its two masked
    incarnations in ``P`` and ``R``, **summed vector-wise**.  The three are paired
    by ``linker_id``, not by position.

``rgroup_contrastive`` [11-12]
    ``z_C = g_Phi(q_i (+) c_R)`` against ``z_R = g_phi(pool(H[R_i]))``.  This is the
    R-group retrieval head that drives lead optimization, and it is **per-linker**:
    one query per detached R-group, not one per molecule.  MolDAM replaced this
    with a sum-pooled R-group bag contrasted against a single core, which collapses
    cardinality; restoring the per-linker form is the point of MolPLAtte.

Stop-gradients
--------------
``sg_P`` / ``sg_R`` / ``sg_Q`` detach the corresponding branch.  The paper applies
``STOPGRAD`` to the **decomposed** branch of both losses 1 and 2 (Eqs. 8 and 10),
i.e. BYOL-style asymmetry, which is the ``sg_Q=True`` default.  MolDAM dropped
these toggles entirely and so has no collapse-mitigation lever.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool

from . import encoders as encoder_registry
from . import heads as head_registry
from .heads import projectors as projector_registry

__all__ = ["MolPLAtteConfig", "MolPLAtte"]

_POOLING = {"mean": global_mean_pool, "add": global_add_pool, "sum": global_add_pool}


@dataclass
class MolPLAtteConfig:
    """Hydra-facing model configuration.

    ``__post_init__`` propagates the shared hyperparameters into each head's
    kwargs wherever the YAML left them ``null``, which is how the config files
    mark "inherit".
    """

    hidden_dim: int = 300
    dropout_rate: float = 0.0
    norm_method: str = "GraphNorm"
    gnn_conv: str = "GINEConv"
    num_conv: int = 5
    graph_pooling: str = "mean"
    condvec_dim: int = 97

    stop_gradient_P: bool = False
    stop_gradient_R: bool = False
    stop_gradient_Q: bool = True

    graph_encoder: str = "VanillaGNN"
    graph_encoder_kwargs: dict = field(default_factory=dict)
    graph_projector: str = "MLPProjector"
    graph_projector_kwargs: dict = field(default_factory=dict)
    node_projector: str = "MLPProjector"
    node_projector_kwargs: dict = field(default_factory=dict)
    query_projector: str = "MLPProjector"
    query_projector_kwargs: dict = field(default_factory=dict)
    rgroup_projector: str = "MLPProjector"
    rgroup_projector_kwargs: dict = field(default_factory=dict)

    #: Assembly head -- recovers the chemistry masking destroyed at each joint,
    #: which is what turns "retrieve this R-group" into "attach it like this".
    #: Off by default so the three MolPLA objectives stay the baseline.
    assembly_head: Optional[str] = None
    assembly_head_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        shared = {
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "norm_method": self.norm_method,
        }
        for name in (
            "graph_encoder_kwargs",
            "graph_projector_kwargs",
            "node_projector_kwargs",
            "query_projector_kwargs",
            "rgroup_projector_kwargs",
            "assembly_head_kwargs",
        ):
            head_kwargs = dict(getattr(self, name) or {})
            for key, value in shared.items():
                if head_kwargs.get(key) is None:
                    head_kwargs[key] = value
            setattr(self, name, head_kwargs)

        self.graph_encoder_kwargs.setdefault("gnn_conv", self.gnn_conv)
        self.graph_encoder_kwargs.setdefault("num_conv", self.num_conv)
        # condvec_dim = 0 removes the condition vector entirely: the query is
        # then the core-side linker node embedding alone. That is what MolDAM
        # does, and it is what makes retrieval a genuine core -> R-group task.
        #
        # MolPLA conditions the query on a functional-group vector OF THE TARGET
        # R-GROUP (paper Eq. 11), so the query carries a description of the
        # answer. Measured on a model trained with it: zeroing the condvec at
        # inference drops hit@1 from 0.6111 to EXACTLY 0.0000 on the queries that
        # had one, and the query projector weights the 97 condvec dimensions
        # 4.09x more per-dimension than the 300 node dimensions. MolPLA's own
        # `Cond. None` ablation collapses its MRR 0.2616 -> 0.0056; a 47x drop
        # from removing an "auxiliary" hint means it was never auxiliary.
        if self.query_projector_kwargs.get("input_dim") is None:
            self.query_projector_kwargs["input_dim"] = self.hidden_dim + self.condvec_dim
        if self.graph_pooling not in _POOLING:
            raise ValueError(
                f"unknown graph_pooling {self.graph_pooling!r}; "
                f"available: {sorted(_POOLING)}"
            )


class MolPLAtte(nn.Module):
    """The composite model. ``forward`` mutates and returns the batch dict."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        known = {f.name for f in fields(MolPLAtteConfig)}
        self.config = MolPLAtteConfig(**{k: v for k, v in kwargs.items() if k in known})
        c = self.config

        self.nnet = nn.ModuleDict()
        self.nnet["graph_encoder"] = getattr(encoder_registry, c.graph_encoder)(
            **c.graph_encoder_kwargs
        )
        self.nnet["graph_projector"] = getattr(projector_registry, c.graph_projector)(
            **c.graph_projector_kwargs
        )
        self.nnet["node_projector"] = getattr(projector_registry, c.node_projector)(
            **c.node_projector_kwargs
        )
        self.nnet["query_projector"] = getattr(projector_registry, c.query_projector)(
            **c.query_projector_kwargs
        )
        self.nnet["rgroup_projector"] = getattr(projector_registry, c.rgroup_projector)(
            **c.rgroup_projector_kwargs
        )
        if c.assembly_head:
            self.nnet["assembly_head"] = getattr(head_registry, c.assembly_head)(
                **c.assembly_head_kwargs
            )
        self.pool = _POOLING[c.graph_pooling]

    @staticmethod
    def _maybe_detach(tensor: torch.Tensor, flag: bool) -> torch.Tensor:
        return tensor.detach() if flag else tensor

    def forward(self, batch: Dict) -> Dict:
        c = self.config

        # --- one encoder pass over every view -------------------------------
        W = self.nnet["graph_encoder"](batch["W"])
        H = W.node_embeddings
        batch["W"] = W

        G_nodes = H[batch["G_markers"]]
        P_nodes = self._maybe_detach(H[batch["P_markers"]], c.stop_gradient_P)
        R_nodes = self._maybe_detach(H[batch["R_markers"]], c.stop_gradient_R)

        node_sample = batch["node_sample"]
        n_samples = int(batch["num_samples"])

        # --- loss 1: graph-level G <-> Q ------------------------------------
        # Q = P u R, pooled per instance. Concatenating the two node sets and
        # their shared instance index pools the union in one call.
        Q_nodes = torch.cat([P_nodes, R_nodes], dim=0)
        Q_index = torch.cat(
            [node_sample[batch["P_markers"]], node_sample[batch["R_markers"]]], dim=0
        )
        z_G = self.nnet["graph_projector"](
            self.pool(G_nodes, node_sample[batch["G_markers"]], size=n_samples)
        )
        z_Q = self.nnet["graph_projector"](self.pool(Q_nodes, Q_index, size=n_samples))
        z_Q = self._maybe_detach(z_Q, c.stop_gradient_Q)

        # --- loss 2: linker-node G <-> (P + R) ------------------------------
        # joint_* index into W, so gather from H directly rather than from the
        # view slices -- no offset arithmetic, no ordering assumption.
        joint_G_idx, joint_P_idx = batch["joint_G_idx"], batch["joint_P_idx"]
        joint_R_idx = batch["joint_R_idx"]
        m_i = H[joint_G_idx]
        q_i = self._maybe_detach(H[joint_P_idx], c.stop_gradient_P)
        r_i = self._maybe_detach(H[joint_R_idx], c.stop_gradient_R)

        z_m = self.nnet["node_projector"](m_i)
        z_p = self.nnet["node_projector"](q_i + r_i)
        z_p = self._maybe_detach(z_p, c.stop_gradient_Q)

        # --- loss 3: R-group retrieval, one query per detached R-group ------
        n_rgroups = int(joint_R_idx.numel())
        use_condvec = c.condvec_dim > 0
        condvec = batch["condvec"].to(H.dtype)
        # The retrieval callbacks build their dedup gallery assuming query and
        # target rows are 1:1 and in the same order. A per-molecule aggregation
        # slipping into either branch would misalign them silently and produce
        # plausible-but-wrong R@K, so state the invariant here rather than trust it.
        if use_condvec and condvec.shape[0] != n_rgroups:
            raise ValueError(
                f"condvec has {condvec.shape[0]} rows but the batch has "
                f"{n_rgroups} detached R-groups"
            )
        if use_condvec and condvec.shape[-1] != c.condvec_dim:
            raise ValueError(
                f"condvec width {condvec.shape[-1]} != configured condvec_dim "
                f"{c.condvec_dim}; the corpus and nnet_module config disagree"
            )
        query_input = torch.cat([q_i, condvec], dim=-1) if use_condvec else q_i
        z_C = self.nnet["query_projector"](query_input)
        rgroup_pooled = self.pool(
            R_nodes, batch["R_pool_index"], size=max(n_rgroups, 1)
        )[:n_rgroups]
        z_R = self.nnet["rgroup_projector"](rgroup_pooled)

        # Assembly runs last: it reads the same joint indices as loss 2 and
        # writes only prediction logits, so it cannot perturb the contrastive
        # branches.
        if "assembly_head" in self.nnet:
            batch = self.nnet["assembly_head"](batch)

        if z_C.shape[0] != z_R.shape[0]:
            raise ValueError(
                f"query/target row mismatch: {z_C.shape[0]} queries vs "
                f"{z_R.shape[0]} R-group embeddings"
            )

        batch.update(
            graph_contrastive=(z_G, z_Q),
            linker_contrastive=(z_m, z_p),
            rgroup_contrastive=(z_C, z_R),
            # Exposed for the retrieval callbacks, L2-NORMALISED.
            #
            # MolDAM normalises inside its retrieval head, so every consumer got
            # unit vectors. MolPLAtte moved normalisation into the loss
            # (DualInfoNCE normalises internally) and the callbacks inherited the
            # old assumption -- FAISSRetrieval and PredictionTable were ranking by
            # RAW inner product, dominated by vector magnitude. Measured on the
            # trained checkpoint: norms span 26-125 (query) and 15-253 (target),
            # giving batch R@1 of 0.0023 against 0.1172 for cosine, with only 43%
            # top-1 agreement between the two rankings.
            #
            # Normalising here restores a single source of truth. The loss's own
            # F.normalize becomes redundant but harmless, and gradients are
            # unchanged: L2 normalisation is idempotent, so re-normalising a unit
            # vector is the identity on its tangent space.
            query_projection=F.normalize(z_C, dim=-1),
            rgroup_projection=F.normalize(z_R, dim=-1),
        )
        return batch

    # ------------------------------------------------------------------ #
    # Library hooks
    #
    # The R-Group Retrieval task scores against a library of every recommendable
    # R-group in the corpus, embedded with the *current* projector.  Because the
    # projector is still training, that library is only valid for the step that
    # built it -- MolPLA rebuilds it at every validation epoch and so does
    # `callbacks/RGroupLibraryRetrieval.py`.  These two hooks are the entry
    # points; keeping them on the model means the callback and the standalone
    # `build_library.py` cannot drift apart in how they embed.
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def encode_rgroups(self, graph_batch) -> torch.Tensor:
        """Embed R-group graphs into the retrieval co-embedding space.

        Parameters
        ----------
        graph_batch
            A PyG ``Batch`` of masked R-group graphs.

        Returns
        -------
        torch.Tensor
            ``(n_graphs, D)``, **not** L2-normalised -- callers normalise once,
            just before building or querying the index, so a caller cannot
            double-normalise without noticing.
        """
        encoded = self.nnet["graph_encoder"](graph_batch)
        pooled = self.pool(encoded.node_embeddings, encoded.batch)
        return self.nnet["rgroup_projector"](pooled)

    @torch.no_grad()
    def encode_queries(
        self, node_embeddings: torch.Tensor, condvec: torch.Tensor
    ) -> torch.Tensor:
        """Project core-side linker nodes (+ condition vector) into the same space.

        This is the inference-time entry point for lead optimization: give it the
        linker node embeddings of a core template and the condition vector of the
        R-group you want, and search the library with the result.
        """
        if condvec.shape[-1] != self.config.condvec_dim:
            raise ValueError(
                f"condvec width {condvec.shape[-1]} != configured condvec_dim "
                f"{self.config.condvec_dim}"
            )
        return self.nnet["query_projector"](
            torch.cat([node_embeddings, condvec.to(node_embeddings.dtype)], dim=-1)
        )
