"""Monitor over-smoothing, over-squashing, over-correlation and over-dilution.

Metric definitions follow Table 1 of Lee et al., "Understanding and Tackling
Over-Dilution in Graph Neural Networks" (KDD 2025). The NATR reference repo
ships the model but no measurement code, so these are implemented from the
paper's formulas.

Three cost tiers, independently gated:

  1. structural  -- MAD, Dirichlet energy, feature correlation, per layer.
                    No autograd; runs every eval epoch.
  2. intra       -- per-attribute dilution of the fusion MLP. One Jacobian
                    per sampled node through a small MLP.
  3. inter/squash-- Jacobian of the conv stack w.r.t. every initial node
                    representation. Yields the inter-node dilution factor and
                    the over-squashing decay curve from one computation.

A note on what tier 2 means for MolPLAtte. The paper assumes h^(0) = sum_t z_t,
which forces delta^intra(t) = 1/|T_v| uniformly. MolPLAtte inherits MolDAM's
VanillaGNN, which concatenates its node-attribute embeddings and learns a
fusion MLP, so the weights *can* be non-uniform -- but they are static, not
context-dependent. Tier 2 therefore measures whether the learned fusion
actually differentiates attributes or collapses toward the uniform 1/|T_v|
baseline.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# NOTE: ``FEAT2DIM_NODE`` / ``FEAT2DIM_EDGE`` are imported lazily inside the
# tiers that need them (tier 2 and tier 3). MolDAM imported them at module
# scope; here that would make ``import callbacks`` depend on nnet_modules,
# which is authored separately. Both call sites already run under the
# per-tier try/except in ``on_validation_batch_end``, so a missing symbol
# degrades to a logged warning instead of an import error.

log = logging.getLogger(__name__)


class RepresentationHealth(pl.Callback):
    """Diagnose representation pathologies in the shared graph encoder."""

    def __init__(self,
                 every_n_epochs:      int = 5,
                 enable_after_epoch:  int = 0,
                 structural_every_epoch: bool = True,
                 n_graphs:            int = 4,
                 n_target_nodes:      int = 2,
                 max_graph_nodes:     int = 48,
                 run_jacobian:        bool = True):
        super().__init__()
        self.every_n_epochs      = int(every_n_epochs)
        self.enable_after_epoch  = int(enable_after_epoch)
        self.structural_every_epoch = bool(structural_every_epoch)
        self.n_graphs            = int(n_graphs)
        self.n_target_nodes      = int(n_target_nodes)
        self.max_graph_nodes     = int(max_graph_nodes)
        self.run_jacobian        = bool(run_jacobian)
        self._done_this_epoch    = False

    # -- plumbing ---------------------------------------------------------

    @property
    def _graph_key(self) -> str:
        # One encoder sees G, P and every R in a single forward pass, so
        # measure it on the largest stream: the intact molecule G. R-groups
        # (a few atoms, diameter ~2) are too small for any of these
        # pathologies to be visible, and P is G minus the detached R-groups.
        return "G"

    @staticmethod
    def _encoder(pl_module) -> torch.nn.Module:
        return pl_module.model.model.nnet["graph_encoder"]

    def _should_run_jacobian(self, trainer) -> bool:
        if not self.run_jacobian:
            return False
        ep = trainer.current_epoch
        return ep >= self.enable_after_epoch and ep % self.every_n_epochs == 0

    # -- tier 1: structural, no autograd -----------------------------------

    @staticmethod
    def _mad(h: torch.Tensor, edge_index: torch.Tensor) -> float:
        """Mean cosine distance across edges. 0 => fully over-smoothed."""
        if edge_index.numel() == 0:
            return float("nan")
        src, dst = edge_index[0], edge_index[1]
        cos = F.cosine_similarity(h[src], h[dst], dim=-1)
        return float((1.0 - cos).mean())

    @staticmethod
    def _dirichlet(h: torch.Tensor, edge_index: torch.Tensor,
                   num_nodes: int) -> float:
        """Sum of squared edge differences / |V| (Table 1). Scale-sensitive:
        read alongside MAD, which is scale-free."""
        if edge_index.numel() == 0:
            return float("nan")
        src, dst = edge_index[0], edge_index[1]
        return float(((h[src] - h[dst]) ** 2).sum() / max(num_nodes, 1))

    @staticmethod
    def _feature_corr(h: torch.Tensor) -> float:
        """Mean |Pearson rho| between feature dimensions (over-correlation)."""
        d = h.size(-1)
        if h.size(0) < 3 or d < 2:
            return float("nan")
        x = h - h.mean(dim=0, keepdim=True)
        std = x.std(dim=0, unbiased=False).clamp_min(1e-12)
        c = (x / std).t() @ (x / std) / x.size(0)
        off = c.abs().sum() - c.diagonal().abs().sum()
        return float(off / (d * (d - 1)))

    def _layerwise(self, enc, graph) -> Dict[str, float]:
        """Tap h^(0)..h^(L) via hooks and compute the scale-free metrics.

        The residual add happens at the *start* of the next block, so the
        input to conv_i is already the post-residual output of block i-1.
        Pre-hooks on the convs therefore give h^(0)..h^(L-1) with no model
        change; graph_conv's own output gives h^(L).
        """
        from torch_geometric.nn import MessagePassing

        # Encoder-agnostic: hook every message-passing layer wherever it lives.
        # VanillaGNN keeps them in one GraphSequential; NatrGNN interleaves them
        # with attribute cross-attention across `node_module_{m}` blocks. Hooks
        # fire in call order, so `taps` is ordered by depth either way.
        taps: List[torch.Tensor] = []
        handles = []
        for mod in enc.modules():
            if isinstance(mod, MessagePassing):
                handles.append(mod.register_forward_pre_hook(
                    lambda m, args, _t=taps: _t.append(args[0].detach())))

        try:
            with torch.no_grad():
                out = enc(graph)
            taps.append(out.node_embeddings.detach())
        finally:
            for h in handles:
                h.remove()

        ei, n = graph.edge_index, graph.num_nodes
        out: Dict[str, float] = {}
        for i, h in enumerate(taps):
            out[f"health/mad/l{i}"]       = self._mad(h, ei)
            out[f"health/dirichlet/l{i}"] = self._dirichlet(h, ei, n)
            out[f"health/corr/l{i}"]      = self._feature_corr(h)
        if len(taps) >= 2:
            m0, mL = out.get("health/mad/l0"), out.get(f"health/mad/l{len(taps)-1}")
            if m0 and m0 > 0 and mL == mL:  # not nan
                out["health/mad/retention"] = mL / m0
        return out

    # -- tier 2: intra-node dilution ---------------------------------------

    def _intra_dilution(self, enc, graph) -> Dict[str, float]:
        """delta^intra(t) per node attribute (Eq. 3).

        h^(0) = fusion_node(concat(z_1..z_6)); attribute t occupies a
        contiguous hidden_dim-wide column block of that input, so the
        influence score I_v(t) is the sum of |J| over that block.

        In eval mode the fusion MLP's norm layers use running statistics, so
        it acts per-node and a single-row Jacobian is exact.
        """
        from torch.func import jacrev

        from nnet_modules.components import FEAT2DIM_NODE

        names = list(FEAT2DIM_NODE.keys())
        hdim  = enc.hidden_dim

        # NATR exposes delta^intra directly: the paper defines it for NATR as
        # the attribute attention coefficient (§6.1), so no Jacobian is needed.
        if getattr(enc, "last_attribute_attention", None) is not None:
            with torch.no_grad():
                enc(graph)
            attn = enc.last_attribute_attention        # (N, n_attr_types)
            if attn is not None and attn.size(1) == len(names):
                delta = attn.mean(dim=0).float()
                delta = (delta / delta.sum().clamp_min(1e-12)).cpu()
                return self._intra_summary(delta, names)

        if "fusion_node" not in enc.moduledict:
            log.info("[RepresentationHealth] encoder has no static fusion_node "
                     "and exposes no attribute attention; skipping intra-dilution")
            return {}

        with torch.no_grad():
            Z = [enc.moduledict[f"embedding_{nf}"](graph[nf].long()) for nf in names]
            C = torch.cat(Z, dim=1)

        fusion = enc.moduledict["fusion_node"]

        def f(c):
            return fusion(c.unsqueeze(0)).squeeze(0)

        n = min(self.max_graph_nodes, C.size(0))
        idx = torch.randperm(C.size(0), device=C.device)[:n]
        scores = torch.zeros(len(names), device=C.device)
        with torch.enable_grad():
            for v in idx.tolist():
                J = jacrev(f)(C[v].detach())            # (hidden, 6*hidden)
                a = J.abs()
                for t in range(len(names)):
                    scores[t] += a[:, t * hdim:(t + 1) * hdim].sum()

        delta = (scores / scores.sum().clamp_min(1e-12)).cpu()
        return self._intra_summary(delta, names)

    @staticmethod
    def _intra_summary(delta: torch.Tensor, names: List[str]) -> Dict[str, float]:
        out = {f"health/intra_dilution/{nf}": float(delta[t])
               for t, nf in enumerate(names)}
        # Normalised entropy: 1.0 == perfectly uniform == maximal intra-node
        # dilution (the 1/|T_v| regime the paper's MPNN baseline is stuck in).
        p = delta.clamp_min(1e-12)
        out["health/intra_dilution/entropy"] = float(
            -(p * p.log()).sum() / np.log(len(names)))
        out["health/intra_dilution/min"] = float(delta.min())
        out["health/intra_dilution/max"] = float(delta.max())
        return out

    # -- tier 3: inter-node dilution + over-squashing -----------------------

    def _inter_and_squash(self, enc, batch_graph) -> Dict[str, float]:
        """delta^inter(v) (Eq. 4) and the over-squashing decay curve.

        Both derive from J_vu = e^T |d h_v^(L) / d h_u^(0)| e, so the Jacobian
        is computed once per sampled target node and consumed twice.
        """
        from torch.func import jacrev

        from nnet_modules.components import FEAT2DIM_EDGE, FEAT2DIM_NODE

        # Needs a clean h^(0) -> h^(L) function, which only the plain
        # fusion_node + graph_conv layout provides. NatrGNN interleaves
        # attribute cross-attention between conv blocks, so the split does not
        # exist there; skip rather than report something mislabelled.
        if not {"fusion_node", "graph_conv"} <= set(enc.moduledict.keys()):
            log.info("[RepresentationHealth] encoder layout has no separable "
                     "h^(0)/conv-stack split; skipping inter-dilution + squash")
            return {}

        n_graphs = min(self.n_graphs, int(batch_graph.num_graphs))
        inter_vals: List[float] = []
        node_counts: List[int] = []
        by_hop: Dict[int, List[float]] = {}

        for gi in range(n_graphs):
            g = batch_graph.get_example(gi)
            N = int(g.num_nodes)
            if N < 3 or N > self.max_graph_nodes or g.edge_index.numel() == 0:
                continue
            node_counts.append(N)

            with torch.no_grad():
                Z = [enc.moduledict[f"embedding_{nf}"](g[nf].long())
                     for nf in FEAT2DIM_NODE]
                h0 = enc.moduledict["fusion_node"](torch.cat(Z, dim=1))
                E = [enc.moduledict[f"embedding_{ef}"](g[ef].long())
                     for ef in FEAT2DIM_EDGE]
                E = enc.moduledict["fusion_edge"](torch.cat(E, dim=1))

            ei = g.edge_index
            bvec = torch.zeros(N, dtype=torch.long, device=h0.device)
            stack = enc.moduledict["graph_conv"]

            # hop distances on the undirected graph
            a = csr_matrix((np.ones(ei.size(1)),
                            (ei[0].cpu().numpy(), ei[1].cpu().numpy())),
                           shape=(N, N))
            spd = shortest_path(a, method="D", directed=False, unweighted=True)

            targets = torch.randperm(N)[:min(self.n_target_nodes, N)].tolist()
            with torch.enable_grad():
                for v in targets:
                    def fv(x, _v=v):
                        return stack(x, ei, edge_attr=E, batch=bvec)[_v]

                    J = jacrev(fv)(h0.detach())         # (hidden, N, hidden)
                    Juv = J.abs().sum(dim=(0, 2))       # (N,)  == e^T|J_vu|e
                    tot = Juv.sum().clamp_min(1e-12)
                    inter_vals.append(float(Juv[v] / tot))

                    share = (Juv / tot).detach().cpu().numpy()
                    for u in range(N):
                        d = spd[v, u]
                        if u == v or not np.isfinite(d):
                            continue
                        by_hop.setdefault(int(d), []).append(float(share[u]))

        out: Dict[str, float] = {}
        if inter_vals:
            out["health/inter_dilution"] = float(np.mean(inter_vals))
        if node_counts:
            out["health/_mean_nodes"] = float(np.mean(node_counts))
        for k in sorted(by_hop):
            if k <= 12:
                out[f"health/squash/h{k}"] = float(np.mean(by_hop[k]))

        # Decay must be read *inside* the receptive field. A graph-aware norm
        # (GraphNorm/LayerNorm/InstanceNorm) couples every node pair through
        # per-graph statistics, so influence beyond `num_conv` hops is
        # normalisation, not message passing -- measured directly: with
        # BatchNorm the share is exactly 0.0 past the receptive field, with
        # GraphNorm it plateaus around 0.05.
        L = int(getattr(enc, "num_conv", 0)) or max(by_hop, default=1)
        in_field = [k for k in by_hop if 1 <= k <= L]
        if 1 in by_hop and len(in_field) > 1:
            edge_hop = max(in_field)
            n1 = np.mean(by_hop[1])
            if n1 > 0:
                out["health/squash/edge_over_near"] = float(np.mean(by_hop[edge_hop]) / n1)
                out["health/squash/receptive_hop"] = float(edge_hop)

        # The graph-norm coupling channel, isolated: mean influence share of
        # nodes strictly outside the receptive field. Should be ~0 for a purely
        # local encoder.
        beyond = [v for k, vs in by_hop.items() if k > L for v in vs]
        if beyond:
            out["health/squash/norm_floor"] = float(np.mean(beyond))
        # Fraction of node pairs the encoder structurally cannot connect.
        total_pairs = sum(len(v) for v in by_hop.values())
        if total_pairs:
            out["health/squash/frac_beyond_field"] = float(len(beyond) / total_pairs)
        return out

    # -- graph-level embedding collapse ------------------------------------

    def _embedding_collapse(self, batch: Dict) -> Dict[str, float]:
        """Collapse diagnostics on the pooled embeddings the heads compare.

        The encoder metrics above are all node-level, but MolDAM's one
        confirmed representation failure (the fragment set-level collapse)
        happened after pooling: batch embeddings became near-parallel, so
        contrastive retrieval had nothing to separate. These two numbers
        detect that directly.

          mean_cos  -- mean off-diagonal cosine over the batch. -> 1 means
                       every embedding points the same way.
          eff_rank  -- exp(entropy of normalised singular values) (Roy &
                       Vetterli). Low relative to dim => dimensional collapse.

        MolPLAtte is *entirely* contrastive -- all three MolPLA losses are
        InfoNCE -- and two of its three branches carry a stop-gradient
        (``sg_Q`` defaults on), which is precisely the SimSiam setting where
        collapse is the named failure mode. These are the load-bearing
        diagnostics here, more so than in MolDAM.
        """
        targets: Dict[str, torch.Tensor] = {}
        # PyG Batch views carrying a pooled ``graph_embedding`` attribute.
        for name in ("G", "P", "Q", "R"):
            g = batch.get(name)
            emb = getattr(g, "graph_embedding", None) if g is not None else None
            if emb is not None:
                targets[name] = emb
        # Flat projection tensors written by the heads.
        for key, label in (("graph_projection_G", "proj_G"),
                           ("graph_projection_Q", "proj_Q"),
                           ("query_projection",   "proj_query"),
                           ("rgroup_projection",  "proj_rgroup")):
            if batch.get(key) is not None:
                targets[label] = batch[key]

        out: Dict[str, float] = {}
        for name, emb in targets.items():
            if emb.dim() != 2 or emb.size(0) < 3:
                continue
            E = emb.detach().float()
            n, d = E.shape
            En = F.normalize(E, dim=-1)
            S = En @ En.t()
            out[f"health/collapse/{name}/mean_cos"] = float(
                (S.sum() - S.diagonal().sum()) / (n * (n - 1)))
            try:
                sv = torch.linalg.svdvals(E - E.mean(dim=0, keepdim=True))
                p = (sv / sv.sum().clamp_min(1e-12)).clamp_min(1e-12)
                er = float(torch.exp(-(p * p.log()).sum()))
                out[f"health/collapse/{name}/eff_rank"] = er
                out[f"health/collapse/{name}/eff_rank_frac"] = er / min(n, d)
            except Exception:
                pass
        return out

    # -- headline scores ----------------------------------------------------

    @staticmethod
    def _headline_scores(m: Dict[str, float]) -> Dict[str, float]:
        """One scalar per pathology, normalised so **higher == worse**.

        Each is anchored to the value the quantity takes in the pathological
        limit, so 0 means healthy and 1 means fully degenerate.
        """
        def clamp(x):
            return float(min(1.0, max(0.0, x)))

        s: Dict[str, float] = {}

        # over-smoothing: MAD collapsing from input to output.
        mad = {int(k.rsplit("l", 1)[1]): v for k, v in m.items()
               if k.startswith("health/mad/l")}
        if len(mad) >= 2:
            m0, mL = mad[min(mad)], mad[max(mad)]
            if m0 > 1e-9:
                s["health/score/oversmoothing"] = clamp(1.0 - mL / m0)

        # over-squashing: influence lost across the receptive field.
        if "health/squash/edge_over_near" in m:
            s["health/score/oversquashing"] = clamp(
                1.0 - m["health/squash/edge_over_near"])

        # intra-node dilution: entropy of the attribute-influence distribution.
        # 1.0 == perfectly uniform == the 1/|T_v| regime; already 0-1.
        if "health/intra_dilution/entropy" in m:
            s["health/score/dilution_intra"] = clamp(
                m["health/intra_dilution/entropy"])

        # inter-node dilution: delta^inter against its uniform floor 1/N,
        # where every node contributes equally to v's final representation.
        d_inter, n_nodes = m.get("health/inter_dilution"), m.get("health/_mean_nodes")
        if d_inter is not None and n_nodes and n_nodes > 1:
            floor = 1.0 / n_nodes
            s["health/score/dilution_inter"] = clamp(
                (1.0 - d_inter) / max(1.0 - floor, 1e-9))

        # over-correlation at the final layer; already 0-1.
        corr = {int(k.rsplit("l", 1)[1]): v for k, v in m.items()
                if k.startswith("health/corr/l")}
        if corr:
            s["health/score/overcorrelation"] = clamp(corr[max(corr)])

        # graph-level collapse: dimensional collapse of the pooled embeddings.
        fracs = [v for k, v in m.items() if k.endswith("/eff_rank_frac")]
        if fracs:
            s["health/score/collapse"] = clamp(1.0 - float(np.mean(fracs)))
        return s

    # -- lightning hooks ----------------------------------------------------

    def on_test_epoch_start(self, trainer, pl_module):
        self.on_validation_epoch_start(trainer, pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch,
                          batch_idx, dataloader_idx=0):
        self.on_validation_batch_end(trainer, pl_module, outputs, batch,
                                     batch_idx, dataloader_idx)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._done_this_epoch = False

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx=0):
        if self._done_this_epoch or trainer.sanity_checking:
            return
        if trainer.current_epoch < self.enable_after_epoch:
            return
        self._done_this_epoch = True

        graph = batch.get(self._graph_key)
        if graph is None:
            log.warning("[RepresentationHealth] no %r in batch; skipping",
                        self._graph_key)
            return

        enc = self._encoder(pl_module)
        was_training = enc.training
        enc.eval()
        metrics: Dict[str, float] = {}
        # Each tier is isolated: one tier failing must not discard the others,
        # nor the headline scores derived from whatever did succeed. PNAConv is
        # the concrete case -- its scatter ops use a C++ autograd::Function that
        # torch.func.jacrev cannot trace, so _inter_and_squash raises while
        # every other tier works fine.
        def _tier(name, fn):
            try:
                metrics.update(fn())
            except Exception as e:
                log.warning("[RepresentationHealth] tier %r skipped: %s: %s",
                            name, type(e).__name__, str(e)[:160])

        try:
            if self.structural_every_epoch:
                _tier("structural", lambda: self._layerwise(enc, graph))
                _tier("collapse",   lambda: self._embedding_collapse(batch))
            if self._should_run_jacobian(trainer):
                # Lightning runs `test` inside torch.inference_mode, where
                # jacrev cannot build a graph. Left alone the tiers do not
                # raise -- they return ZEROS, which log as intra-H 0.000 and
                # dilution_inter 1.000 and read exactly like a measurement of a
                # fully degenerate representation. The same checkpoint on the
                # same corpus reports 0.987 / 0.221 from the validation loop.
                # So: re-enable grad, and if that is impossible say so rather
                # than emit a number.
                if torch.is_inference_mode_enabled():
                    log.warning(
                        "[RepresentationHealth] Jacobian tiers SKIPPED: running "
                        "under torch.inference_mode (trainer.test), where "
                        "jacrev silently yields zeros. Structural metrics below "
                        "are unaffected; dilution/over-squashing are NOT "
                        "reported rather than reported as zero.")
                else:
                    with torch.enable_grad():
                        _tier("intra_dilution",
                              lambda: self._intra_dilution(enc, graph))
                        _tier("inter_squash",
                              lambda: self._inter_and_squash(enc, graph))
            _tier("scores", lambda: self._headline_scores(metrics))
        finally:
            if was_training:
                enc.train()

        # drop NaN and private keys (leading underscore in the last segment)
        clean = {k: v for k, v in metrics.items()
                 if v == v and not k.rsplit("/", 1)[-1].startswith("_")}
        for k, v in clean.items():
            pl_module.log(k, v, on_step=False, on_epoch=True, batch_size=1)
        if clean:
            self._log_summary(clean)

    @staticmethod
    def _log_summary(m: Dict[str, float]) -> None:
        mad = {int(k.rsplit("l", 1)[1]): v for k, v in m.items()
               if k.startswith("health/mad/l")}
        parts = []
        if mad:
            seq = " -> ".join(f"{mad[i]:.3f}" for i in sorted(mad))
            parts.append(f"MAD {seq}")
        if "health/intra_dilution/entropy" in m:
            parts.append(f"intra-H {m['health/intra_dilution/entropy']:.3f}")
        if "health/inter_dilution" in m:
            parts.append(f"inter {m['health/inter_dilution']:.4f}")
        if parts:
            log.info("[RepresentationHealth] " + " | ".join(parts))
        scores = {k.rsplit("/", 1)[1]: v for k, v in m.items()
                  if k.startswith("health/score/")}
        if scores:
            log.info("[RepresentationHealth] scores (higher=worse) " +
                     "  ".join(f"{k}={v:.3f}" for k, v in sorted(scores.items())))
