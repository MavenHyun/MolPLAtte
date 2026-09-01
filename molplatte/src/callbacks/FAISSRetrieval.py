from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class FAISSRetrieval(pl.Callback):
    """Full-corpus R-group retrieval, scored the way MolPLA's loss #3 is trained.

    Batch contract (produced by the model's ``forward``, consumed here)
    ---------------------------------------------------------------------
    ``batch["query_projection"]``   (n_R, D)  one row per **detached R-group**:
                                    the core-side linker node embedding of ``P``
                                    concatenated with the condition vector and
                                    pushed through the query projector.
    ``batch["rgroup_projection"]``  (n_R, D)  pooled ``R`` graph embedding for
                                    the same rows, in the same order.
    ``batch["R_hashes"]``           flat ``list[str]`` of length ``n_R`` -- the
                                    WL subgraph hash of each detached R-group.

    Note the row axis is *per linker*, not per molecule: MolPLA's retrieval head
    is per-linker (paper §2.3), which is exactly the trait MolDAM dropped by
    sum-pooling R-groups into one bag per core. There is therefore no
    ``index_add_`` aggregation step here -- the pairing is already 1:1.
    """

    KS: Tuple[int, ...] = (1, 5, 10, 100)

    def __init__(self,
                 # "flat_cpu" by default, not "flat_gpu": faiss-gpu-cu12 1.14.1
                 # ships no kernels for this host's RTX PRO 6000 (Blackwell,
                 # sm_120) and dies with "no kernel image is available for
                 # execution on the device" the first time a validation epoch
                 # builds an index. The gallery here is thousands of rows, not
                 # millions, so exact CPU search costs little.
                 index_type: str = "flat_cpu",
                 hnsw_m:     int = 32,
                 enable_after_epoch: int = 0,
                 popularity_correction: bool = False,
                 temperature: float = 0.01):
        """
        popularity_correction
            Rank by ``sim/temperature + log p(k)`` instead of ``sim``.

            Set True for checkpoints trained WITHOUT the loss-side logQ
            correction: those estimate PMI (``log p(k|q) - log p(k)``), so the
            popularity term has to be added back to recover the posterior that
            r@k rewards. Measured on MolDAM v29a/GINEConv (ZINC anchored):
            r@10 0.3805 -> 0.6646. MolPLAtte's corpus and vocabulary differ,
            so that magnitude does not transfer -- the *mechanism* does, and
            any skewed R-group vocabulary reproduces it.

            Leave False for models trained WITH logq_correction=True -- they
            already estimate the posterior, and correcting again double-counts.
            Both variants are logged either way (``faiss/r@k`` and
            ``faiss/uncorrected_r@k``) so a mismatch is visible rather than
            silent.
        """
        super().__init__()
        self.index_type = index_type
        self.hnsw_m     = hnsw_m
        self.enable_after_epoch = int(enable_after_epoch)
        self.popularity_correction = bool(popularity_correction)
        self.temperature = float(temperature)
        self._q:  List[torch.Tensor] = []
        self._g:  List[torch.Tensor] = []
        # Chemical identity per query, so retrieving a *duplicate* of the
        # target counts as a hit rather than an error. See _compute_and_log.
        self._keys: List[str] = []

    def _gated(self, trainer) -> bool:
        return trainer.current_epoch < self.enable_after_epoch

    def _snapshot(self, batch: Dict) -> None:
        q = batch.get("query_projection")
        g = batch.get("rgroup_projection")
        if q is None or g is None:
            return
        self._q.append(q.detach().cpu())
        self._g.append(g.detach().cpu())

        # One hash per row -- unlike MolDAM's anchored batch, where the key was
        # a comma-joined *set* of R-group hashes per core. Per-linker retrieval
        # makes the key a single R-group, which is also the multi-positive
        # grouping key the loss uses.
        hashes = batch.get("R_hashes")
        if hashes is not None:
            self._keys.extend(str(h) for h in hashes)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self._gated(trainer):
            return
        self._snapshot(batch)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._snapshot(batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or self._gated(trainer):
            self._q.clear(); self._g.clear(); self._keys.clear()
            return
        self._compute_and_log(pl_module, stage="val")
        self._q.clear(); self._g.clear(); self._keys.clear()

    def on_test_epoch_end(self, trainer, pl_module):
        self._compute_and_log(pl_module, stage="test")
        self._q.clear(); self._g.clear(); self._keys.clear()

    def _build_index(self, vectors: "np.ndarray"):
        import faiss
        D = vectors.shape[1]
        if self.index_type == "flat_cpu":
            idx = faiss.IndexFlatIP(D)
        elif self.index_type == "flat_gpu":
            res = faiss.StandardGpuResources()
            idx = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(D))
        elif self.index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(D, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown FAISSRetrieval index_type: {self.index_type!r}")
        idx.add(vectors)
        return idx

    def _compute_and_log(self, pl_module: pl.LightningModule, stage: str) -> None:
        if not self._q:
            return

        q = torch.cat(self._q).float().numpy().astype("float32")
        g = torch.cat(self._g).float().numpy().astype("float32")

        valid = (np.linalg.norm(q, axis=1) > 1e-6) & (np.linalg.norm(g, axis=1) > 1e-6)
        q = q[valid]; g = g[valid]
        N = q.shape[0]
        if N < 2:
            return

        # Chemical-identity group per row. Duplicates of the target are correct
        # retrievals, not errors: index-matched scoring caps a perfect model
        # well below 1.0 because a single R-group can occupy a large share of
        # the corpus. gid=None => fall back to index matching.
        gid = None
        if len(self._keys) == valid.shape[0]:
            uniq: dict = {}
            gid = np.fromiter(
                (uniq.setdefault(k, len(uniq)) for k in self._keys),
                dtype=np.int64, count=len(self._keys))[valid]

        idx_q = self._build_index(q)
        idx_g = self._build_index(g)

        max_k = min(max(self.KS), N)
        _, top_qg = idx_g.search(q, max_k)
        _, top_gq = idx_q.search(g, max_k)
        labels    = np.arange(N)

        # ------------------------------------------------------------------
        # Deduplicated gallery: index one representative per distinct
        # chemistry, not one per entry. On the entry gallery a single popular
        # R-group holds a large fraction of rows, so once the top-ranked
        # chemistry is picked its duplicates fill every remaining slot --
        # correct nearest-neighbour behaviour, but it makes r@k collapse onto
        # r@1 and understates r@10. Measured on MolDAM's first_run:
        # r@10 0.2382 -> 0.3958, r@100 0.2793 -> 0.6109. MolPLAtte's own skew is
        # its own; the failure mode is not.
        #
        # Direction note: only query -> chemistry is well posed. The reverse
        # (chemistry -> which of its many linker queries) has no single right
        # answer, so the primary metric is forward-only. strict_r@k keeps the
        # old symmetric index-matched definition for continuity.
        # ------------------------------------------------------------------
        top_dedup, K_dedup = None, None
        top_corr = None
        if gid is not None:
            K_dedup = int(gid.max()) + 1
            first = np.zeros(K_dedup, dtype=np.int64)
            seen = np.zeros(K_dedup, dtype=bool)
            for i, gg in enumerate(gid):
                if not seen[gg]:
                    first[gg] = i
                    seen[gg] = True
            gd = np.ascontiguousarray(g[first])
            idx_dedup = self._build_index(gd)
            _, top_dedup = idx_dedup.search(q, min(max(self.KS), K_dedup))
            pl_module.log(f"{stage}/faiss/gallery_size", float(K_dedup),
                          on_epoch=True, sync_dist=False)

            # Popularity-corrected ranking. The gallery is small after dedup,
            # so score in query chunks rather than materialising an N x K
            # matrix, then take top-k of the corrected score.
            if self.popularity_correction:
                counts = np.bincount(gid, minlength=K_dedup).astype(np.float64)
                logp = np.log(np.maximum(counts, 1.0) / counts.sum()).astype("float32")
                kmax = min(max(self.KS), K_dedup)
                chunks = []
                for s0 in range(0, q.shape[0], 4096):
                    sc = q[s0:s0 + 4096] @ gd.T / self.temperature + logp[None, :]
                    chunks.append(np.argpartition(-sc, kmax - 1, axis=1)[:, :kmax])
                    # argpartition is unordered; sort the retained slice
                    part = chunks[-1]
                    rows = np.arange(part.shape[0])[:, None]
                    order = np.argsort(-sc[rows, part], axis=1)
                    chunks[-1] = part[rows, order]
                top_corr = np.concatenate(chunks, axis=0)

        for k in self.KS:
            if N < k:
                continue
            # strict = index-matched on the entry gallery (old definition)
            s_qg = (top_qg[:, :k] == labels[:, None]).any(1).mean()
            s_gq = (top_gq[:, :k] == labels[:, None]).any(1).mean()
            pl_module.log(f"{stage}/faiss/strict_r@{k}",
                          float((s_qg + s_gq) / 2.0),
                          on_epoch=True, sync_dist=False)

            # entry gallery, collision-aware
            if gid is None:
                hit_qg, hit_gq = s_qg, s_gq
            else:
                hit_qg = (gid[top_qg[:, :k]] == gid[:, None]).any(1).mean()
                hit_gq = (gid[top_gq[:, :k]] == gid[:, None]).any(1).mean()
            pl_module.log(f"{stage}/faiss/entry_r@{k}",
                          float((hit_qg + hit_gq) / 2.0),
                          on_epoch=True, sync_dist=False)

            # PRIMARY: deduplicated gallery, query -> chemistry
            if top_dedup is not None and k <= K_dedup:
                raw = (top_dedup[:, :k] == gid[:, None]).any(1).mean()
            else:
                raw = float((hit_qg + hit_gq) / 2.0)
            # Always log the uncorrected number so a correction mismatch shows
            # up as a gap rather than as a silently wrong headline.
            pl_module.log(f"{stage}/faiss/uncorrected_r@{k}", float(raw),
                          on_epoch=True, sync_dist=False)
            if top_corr is not None and k <= K_dedup:
                hit = (top_corr[:, :k] == gid[:, None]).any(1).mean()
            else:
                hit = raw
            pl_module.log(f"{stage}/faiss/r@{k}", float(hit),
                          on_epoch=True, sync_dist=False)

        pl_module.log(f"{stage}/faiss/mrr",
                      0.5 * (_mrr(top_qg, labels) + _mrr(top_gq, labels)),
                      on_epoch=True, sync_dist=False)
        pl_module.log(f"{stage}/faiss/n_queries", float(N),
                      on_epoch=True, sync_dist=False)
        k10 = min(10, max_k)
        if top_dedup is not None:
            # Report the SAME ranking that faiss/r@k logs. Reading top_dedup
            # here while the metric used top_corr made the console line (which
            # the monitor and leaderboard parse) report uncorrected numbers for
            # popularity-corrected runs -- and inverted a MolDAM A/B conclusion.
            src = top_corr if top_corr is not None else top_dedup
            r1  = (src[:, 0] == gid).mean()
            r10 = (src[:, :min(10, K_dedup)] == gid[:, None]).any(1).mean()
            tag = (f"gallery={K_dedup}"
                   + ("+logp" if top_corr is not None else ""))
        else:
            r1  = (top_qg[:, 0] == labels).mean()
            r10 = (top_qg[:, :k10] == labels[:, None]).any(1).mean()
            tag = "gallery=entries"
        s1  = (top_qg[:, 0] == labels).mean()
        s10 = (top_qg[:, :k10] == labels[:, None]).any(1).mean()
        logging.getLogger(__name__).info(
            f"[FAISSRetrieval/{stage}] N={N} {tag}  "
            f"R@1={r1:.3f}  R@10={r10:.3f}  "
            f"(strict R@1={s1:.3f} R@10={s10:.3f})")


def _mrr(top: np.ndarray, labels: np.ndarray) -> float:
    hits  = (top == labels[:, None])
    pos   = hits.argmax(1)
    in_k  = hits.any(1)
    rr    = np.where(in_k, 1.0 / (pos + 1), 0.0)
    return float(rr.mean())
