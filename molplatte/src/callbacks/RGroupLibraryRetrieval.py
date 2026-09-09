"""Full-library R-group retrieval — MolPLA's actual RGR evaluation.

What this measures, and why the in-batch version is not the same thing
---------------------------------------------------------------------
MolPLA's R-Group Retrieval task embeds **every recommendable R-group in the
corpus** (61,279 of them on GEOM) with the current R-group projector, indexes them
with FAISS, and retrieves the top 1000 per query.  MRR and Hit@K are computed over
that library.  ``FAISSRetrieval`` in this repo scores against a val-split gallery
of a few thousand rows instead, which is a materially easier problem reported
under a similar name.  Both are logged, under distinct prefixes:

* ``{stage}/faiss/*``   — val-split gallery (cheap, every epoch)
* ``{stage}/library/*`` — full corpus library (this callback)

The library is rebuilt every ``every_n_epochs`` because the projector is still
training: a library embedded at epoch 3 is meaningless for a query embedded at
epoch 7.  Rebuild cost is one encoder pass over the vocabulary.

Reading the numbers
-------------------
Hit@K against a frequency-skewed library is *mostly* a measurement of the
frequency prior.  The FlavorDB macfrag vocabulary has ~1.7K distinct R-groups but
an **effective size of 23** — one R-group is 21% of all occurrences.  So this
callback logs, alongside every Hit@K, the score of the constant predictor that
always returns the K most frequent R-groups:

* ``library/hit@K``        — the model
* ``library/prior_hit@K``  — always-return-the-K-most-frequent
* ``library/lift@K``       — the ratio

MolDAM's headline retrieval figure was originally framed as "47x above random";
re-scored against the frequency prior it landed *at* the prior.  Random is the
wrong reference. ``lift@K <= 1`` means the model has learned nothing the prior
does not already give you, whatever ``hit@K`` says.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from retrieval_scoring import logq_corrected, rgroup_temperature
import pytorch_lightning as pl
import torch

logger = logging.getLogger(__name__)

__all__ = ["RGroupLibraryRetrieval"]

#: MolPLA searches the top 1000 and reports these cut-offs.
_KS = (1, 5, 10, 20, 50, 100, 500, 1000)


class RGroupLibraryRetrieval(pl.Callback):
    """Rebuild the R-group vector library each validation epoch and score against it.

    Parameters
    ----------
    vocab_path
        Path to the vocabulary built by ``enumerate_rgroups.py``.  If it does not
        exist the callback disables itself with a warning rather than failing the
        run — a missing library should not cost you a pretraining job.
    max_entries
        Truncate the library to its N most frequent R-groups.  ``None`` uses all of
        them.  Truncation is logged, because a smaller library inflates Hit@K.
    every_n_epochs, enable_after_epoch
        Rebuild cadence and warm-up gate.  Retrieval over random embeddings is
        noise, and the rebuild is the expensive part of the epoch.
    search_k
        Top-K retrieved per query; MolPLA uses 1000.
    encode_batch_size
        Graphs per forward pass when embedding the library.
    max_queries
        Cap on validation queries scored per epoch, subsampled with a fixed seed
        so the estimate is comparable across epochs.  Exact search is
        ``n_queries x n_rows x dim``: on COCONUT that is ~50K queries against a
        ~100K-row library, which costs minutes per epoch on CPU and would dominate
        validation.  ``None`` scores everything.  Whenever the cap bites it is
        logged and surfaced as ``library/queries_truncated`` -- a silently
        subsampled metric reads as a full-corpus number when it is not.
    """

    def __init__(
        self,
        vocab_path: Optional[str] = None,
        max_entries: Optional[int] = None,
        every_n_epochs: int = 1,
        enable_after_epoch: int = 2,
        search_k: int = 1000,
        encode_batch_size: int = 1024,
        max_queries: Optional[int] = 20000,
        temperature: Optional[float] = None,
        popularity_coef: float = 1.0,
    ) -> None:
        super().__init__()
        self.vocab_path = vocab_path
        self.max_entries = max_entries
        self.every_n_epochs = max(1, every_n_epochs)
        self.enable_after_epoch = enable_after_epoch
        self.search_k = search_k
        self.encode_batch_size = encode_batch_size
        self.max_queries = max_queries
        # Read from the training config rather than restated here, so this and
        # LeadOptimizer cannot drift apart. Pass a float only to deliberately
        # score at a temperature the model was NOT trained at.
        self.temperature = (rgroup_temperature() if temperature is None
                            else float(temperature))
        # MolDAM found the optimum at exactly the theoretically predicted 1.0.
        self.popularity_coef = popularity_coef

        self._vocab = None
        self._log_prior = None
        self._disabled = False
        self._queries: List[torch.Tensor] = []
        self._target_hashes: List[str] = []

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _load_vocab(self, trainer: "pl.Trainer") -> bool:
        if self._vocab is not None:
            return True
        if self._disabled:
            return False
        try:
            from data_modules.rgroup_vocab import RGroupLibraryVocab

            path = self.vocab_path
            if path is None:
                # Default: sitting next to the corpus the DataModule is reading.
                corpus = Path(trainer.datamodule.config.dataset_path)
                path = corpus / "rgroup_vocab.pkl.gz"
            self._vocab = RGroupLibraryVocab(path, max_entries=self.max_entries)
        except Exception as exc:  # noqa: BLE001 -- never fail a run over evaluation
            logger.warning(
                "[RGroupLibraryRetrieval] disabled: %s. Build the vocabulary with "
                "`python enumerate_rgroups.py --corpus <corpus>` to enable "
                "full-library retrieval metrics.",
                exc,
            )
            self._disabled = True
            return False

        # log p(k) over library rows, for the scoring-time correction above.
        prior = np.asarray(self._vocab.frequency_prior, dtype=np.float64)
        self._log_prior = np.log(np.clip(prior, 1e-12, None)).astype(np.float32)

        logger.info(
            "[RGroupLibraryRetrieval] library: %s rows, effective size %.0f, "
            "top-1 share %.2f%%",
            f"{len(self._vocab):,}",
            self._vocab.effective_size(),
            100.0 * self._vocab.frequency_prior[0],
        )
        return True

    def _active(self, trainer: "pl.Trainer") -> bool:
        if self._disabled or trainer.sanity_checking:
            return False
        # A test run is a single pass with no epoch schedule: current_epoch is
        # whatever the checkpoint left behind, so the every_n/enable_after
        # gating below is meaningless and would silently skip the evaluation.
        if trainer.testing:
            return True
        epoch = trainer.current_epoch
        if epoch < self.enable_after_epoch:
            return False
        return (epoch - self.enable_after_epoch) % self.every_n_epochs == 0

    # ------------------------------------------------------------------ #
    # Accumulate queries
    # ------------------------------------------------------------------ #
    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._queries.clear()
        self._target_hashes.clear()

    # Test hooks delegate: the body is already stage-aware (`stage = "test" if
    # trainer.testing else "val"`), it simply was never invoked on a test pass.
    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self.on_validation_epoch_start(trainer, pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                          dataloader_idx=0) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch,
                                     batch_idx, dataloader_idx)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self.on_validation_epoch_end(trainer, pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if not self._active(trainer):
            return
        query = batch.get("query_projection")
        hashes = batch.get("R_hashes")
        if query is None or not hashes or query.shape[0] != len(hashes):
            return
        self._queries.append(query.detach().float().cpu())
        self._target_hashes.extend(hashes)

    # ------------------------------------------------------------------ #
    # Rebuild + score
    # ------------------------------------------------------------------ #
    @torch.no_grad()

    def _novel_row_mask(self):
        """Boolean over library rows: True where the R-group is NOT in the base
        vocabulary. ``None`` unless the loaded library is a union build."""
        if getattr(self, "_novel_rows", None) is not None:
            return self._novel_rows
        prov = getattr(self._vocab, "provenance", None) or {}
        novel = prov.get("novel_hashes")
        if not novel:
            self._novel_rows = None
            return None
        novel = set(novel)
        self._novel_rows = np.array([h in novel for h in self._vocab.hashes], dtype=bool)
        logger.info(
            "[RGroupLibraryRetrieval] union library: %s of %s rows are novel "
            "(absent from the base vocabulary); hit@K reported split",
            f"{int(self._novel_rows.sum()):,}", f"{len(self._novel_rows):,}",
        )
        return self._novel_rows

    def _build_library(self, pl_module) -> Optional[np.ndarray]:
        model = pl_module.model.model  # LightningModule -> LossModule -> nnet
        was_training = model.training
        model.eval()
        try:
            chunks = []
            for graph_batch in self._vocab.batches(self.encode_batch_size):
                graph_batch = graph_batch.to(pl_module.device)
                chunks.append(
                    model.encode_rgroups(graph_batch).detach().float().cpu()
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RGroupLibraryRetrieval] library build failed: %s", exc)
            return None
        finally:
            model.train(was_training)
        if not chunks:
            return None
        library = torch.cat(chunks, dim=0)
        library = torch.nn.functional.normalize(library, dim=-1)
        return library.numpy().astype(np.float32)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._active(trainer) or not self._queries:
            return
        if not self._load_vocab(trainer):
            return

        library = self._build_library(pl_module)
        if library is None:
            return

        queries = torch.cat(self._queries, dim=0)
        queries = torch.nn.functional.normalize(queries, dim=-1).numpy().astype(np.float32)
        target_rows = self._vocab.rows_of(self._target_hashes)

        # Queries whose true R-group is not in the library cannot be scored --
        # count them rather than letting them dilute the metric silently.
        in_library = target_rows >= 0
        n_missing = int((~in_library).sum())
        if in_library.sum() == 0:
            logger.warning(
                "[RGroupLibraryRetrieval] no validation target is present in the "
                "library; is the vocabulary from a different corpus?"
            )
            return
        queries = queries[in_library]
        target_rows = target_rows[in_library]

        n_scorable = queries.shape[0]
        truncated = 0
        if self.max_queries is not None and n_scorable > self.max_queries:
            # Fixed seed: the subsample must not wander between epochs or the
            # trajectory would mix real learning with sampling noise.
            rng = np.random.default_rng(0)
            keep = rng.choice(n_scorable, size=self.max_queries, replace=False)
            queries = queries[keep]
            target_rows = target_rows[keep]
            truncated = n_scorable - self.max_queries

        try:
            from .exact_search import search as exact_search

            k = min(self.search_k, library.shape[0])
            # Exact inner product on the GPU. faiss-gpu has no sm_120 kernels,
            # so its GPU index aborts the process on Blackwell and its CPU index
            # takes 171.8s on this shape against 0.4s here -- same arithmetic,
            # 100% top-10 agreement, not an ANN approximation.
            scores, retrieved = exact_search(queries, library, k,
                                             device=str(pl_module.device))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RGroupLibraryRetrieval] search failed: %s", exc)
            return

        # ---- popularity-corrected ranking (MolDAM devlog Phase 10) -----------
        # InfoNCE's Bayes-optimal critic is f*(q,k) = log p(k|q) - log p(k): it
        # deliberately DISCOUNTS popularity, estimating PMI rather than the
        # posterior. hit@K rewards the posterior. With a skewed p(k) -- ours has
        # effective vocabulary 32 -- those orderings diverge, so ranking a
        # PMI-trained model by pure similarity and comparing it to a
        # posterior-optimal frequency prior is not a fair test.
        #
        # MolDAM drew the conclusion "no configuration beats the frequency prior"
        # across 7 checkpoints, 6 architectures and 2 losses on exactly that
        # mistake, then overturned it in Phase 10: adding log p back at SCORING
        # time took their r@10 from 0.3805 (below a 0.4974 prior) to 0.6646,
        # i.e. 1.34x the prior, with the optimum at the theoretically predicted
        # coefficient of 1.0.
        #
        # Both rankings are logged. `hit@K` stays pure-similarity so it remains
        # comparable to earlier runs; `corrected_hit@K` is the one to read.
        corrected = None
        if self._log_prior is not None:
            # CAVEAT: this re-ranks within the similarity top-k, not the whole
            # library, so an item the correction would promote from outside the
            # top-k cannot be recovered. With search_k=1000 that mostly affects
            # hit@1000; hit@1..100 are essentially unaffected.
            adj = logq_corrected(scores, self._log_prior[retrieved],
                                 self.temperature, self.popularity_coef)
            order = np.argsort(-adj, axis=1)
            corrected = np.take_along_axis(retrieved, order, axis=1)

        stage = "test" if trainer.testing else "val"

        hits = retrieved == target_rows[:, None]
        # Rank of the true R-group; queries that miss the top-K score 0 reciprocal.
        has_hit = hits.any(axis=1)
        first_hit = np.where(has_hit, hits.argmax(axis=1) + 1, 0)
        mrr = float(np.where(has_hit, 1.0 / np.maximum(first_hit, 1), 0.0).mean())

        metrics: Dict[str, float] = {
            f"{stage}/library/mrr": mrr,
            f"{stage}/library/size": float(library.shape[0]),
            f"{stage}/library/effective_size": self._vocab.effective_size(),
            f"{stage}/library/n_queries": float(queries.shape[0]),
            f"{stage}/library/n_target_missing": float(n_missing),
            f"{stage}/library/queries_truncated": float(truncated),
        }
        # ---- long-tail diagnostics ------------------------------------------
        # hit@K is a MICRO average, and this task's target distribution is
        # savagely long-tailed: 52.6% of true targets are among the 10 most
        # frequent R-groups. So a headline hit@1 is dominated by an easy majority
        # class and can look strong while the model has learned little beyond the
        # prior. Measured on the v2 checkpoint: micro hit@1 0.712 against a MACRO
        # (frequency-bucket-balanced) 0.472, and accuracy falling 0.924 -> 0.130
        # from the head bucket to the tail.
        #
        # Two further symptoms, both logged:
        #   rank1_distinct  how many different R-groups ever appear at rank 1.
        #                   Measured 680 out of a 48,671-row library; for queries
        #                   whose condition vector is all-zero it collapses to 41.
        #   coverage        fraction of the library ever appearing in any top-K
        #                   (measured 11.4%).
        counts = getattr(self._vocab, "counts", None)
        if counts is not None and len(counts):
            order = np.argsort(-counts)
            freq_pos = np.empty(len(counts), dtype=np.int64)
            freq_pos[order] = np.arange(len(counts))
            tpos = freq_pos[target_rows]
            top1 = retrieved[:, 0] == target_rows
            bucket_acc = []
            for lo, hi, lab in ((0, 10, "0_10"), (10, 100, "10_100"),
                                (100, 1000, "100_1k"), (1000, 10000, "1k_10k"),
                                (10000, 1 << 30, "10k_plus")):
                sel = (tpos >= lo) & (tpos < hi)
                if sel.sum() == 0:
                    continue
                acc = float(top1[sel].mean())
                metrics[f"{stage}/library/hit@1/freq_{lab}"] = acc
                bucket_acc.append(acc)
            if bucket_acc:
                # Unweighted over buckets: the number that does NOT let the
                # frequent head carry the score.
                metrics[f"{stage}/library/macro_hit@1"] = float(np.mean(bucket_acc))
            metrics[f"{stage}/library/rank1_distinct"] = float(len(set(retrieved[:, 0].tolist())))
            metrics[f"{stage}/library/coverage"] = float(
                len(set(retrieved.flatten().tolist())) / max(library.shape[0], 1)
            )
        if corrected is not None:
            chits = corrected == target_rows[:, None]
            chas = chits.any(axis=1)
            cfirst = np.where(chas, chits.argmax(axis=1) + 1, 0)
            metrics[f"{stage}/library/corrected_mrr"] = float(
                np.where(chas, 1.0 / np.maximum(cfirst, 1), 0.0).mean()
            )

        # ---- base vs novel R-groups (union libraries) -----------------------
        # When the library is a UNION of the pretraining vocabulary and a new
        # corpus (see scripts/build_union_vocab.py), a single averaged hit@K
        # hides the question the pocket stage exists to answer: does the model
        # retrieve R-groups it never saw in pretraining, or only the familiar
        # ones? 40% of CrossDocked's distinct R-groups are novel, but they are
        # RARE, so they contribute little to a micro average and the headline
        # number would stay comfortable while novel retrieval failed outright.
        novel_mask = self._novel_row_mask()
        if novel_mask is not None:
            is_novel = novel_mask[target_rows]
            metrics[f"{stage}/library/n_queries_novel"] = float(is_novel.sum())
            metrics[f"{stage}/library/n_queries_base"] = float((~is_novel).sum())

        summary = []
        for cut in _KS:
            if cut > k:
                continue
            hit = float(hits[:, :cut].any(axis=1).mean())
            prior = self._vocab.prior_hit_at_k(target_rows, cut)
            metrics[f"{stage}/library/hit@{cut}"] = hit
            metrics[f"{stage}/library/prior_hit@{cut}"] = prior
            if novel_mask is not None:
                for sub, m in (("novel", is_novel), ("base", ~is_novel)):
                    if m.any():
                        metrics[f"{stage}/library/{sub}_hit@{cut}"] = float(
                            hits[m][:, :cut].any(axis=1).mean()
                        )
            if corrected is not None:
                chit = float(chits[:, :cut].any(axis=1).mean())
                metrics[f"{stage}/library/corrected_hit@{cut}"] = chit
                metrics[f"{stage}/library/corrected_lift@{cut}"] = (
                    chit / prior if prior > 0 else float("nan")
                )
            # lift <= 1 means the model adds nothing over "return the most common".
            metrics[f"{stage}/library/lift@{cut}"] = (
                hit / prior if prior > 0 else float("nan")
            )
            if cut in (1, 10, 100):
                summary.append(f"H@{cut}={hit:.4f}(prior {prior:.4f})")
                # The split goes on the CONSOLE line, not only into log_dict.
                # The log file is what survives a run and what gets read months
                # later; a metric that exists only in the logger is invisible to
                # anyone reading the artifact. And this is the metric that says
                # whether the pocket stage generalised at all -- a healthy
                # hit@K beside a collapsed novel_hit@K means the model retrieved
                # chemistry it already knew.
                if novel_mask is not None:
                    for sub in ("base", "novel"):
                        key = f"{stage}/library/{sub}_hit@{cut}"
                        if key in metrics:
                            summary.append(f"{sub}_hit@{cut}={metrics[key]:.4f}")

        pl_module.log_dict(metrics, on_epoch=True, sync_dist=True)
        logger.info(
            "[RGroupLibraryRetrieval/%s] library=%s eff=%.0f N=%s MRR=%.4f %s%s",
            stage,
            f"{library.shape[0]:,}",
            self._vocab.effective_size(),
            f"{queries.shape[0]:,}",
            mrr,
            "  ".join(summary),
            (
                (f"  (skipped {n_missing:,} out-of-library targets)" if n_missing else "")
                + (
                    f"  (subsampled from {n_scorable:,} queries)"
                    if truncated
                    else ""
                )
            ),
        )
