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
    ) -> None:
        super().__init__()
        self.vocab_path = vocab_path
        self.max_entries = max_entries
        self.every_n_epochs = max(1, every_n_epochs)
        self.enable_after_epoch = enable_after_epoch
        self.search_k = search_k
        self.encode_batch_size = encode_batch_size
        self.max_queries = max_queries

        self._vocab = None
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
            import faiss

            index = faiss.IndexFlatIP(library.shape[1])
            index.add(library)
            k = min(self.search_k, library.shape[0])
            _scores, retrieved = index.search(queries, k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RGroupLibraryRetrieval] FAISS search failed: %s", exc)
            return

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
        summary = []
        for cut in _KS:
            if cut > k:
                continue
            hit = float(hits[:, :cut].any(axis=1).mean())
            prior = self._vocab.prior_hit_at_k(target_rows, cut)
            metrics[f"{stage}/library/hit@{cut}"] = hit
            metrics[f"{stage}/library/prior_hit@{cut}"] = prior
            # lift <= 1 means the model adds nothing over "return the most common".
            metrics[f"{stage}/library/lift@{cut}"] = (
                hit / prior if prior > 0 else float("nan")
            )
            if cut in (1, 10, 100):
                summary.append(f"H@{cut}={hit:.4f}(prior {prior:.4f})")

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
