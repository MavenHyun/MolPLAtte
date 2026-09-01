"""Training-side view of the R-group library vocabulary.

The vocabulary is built once per corpus by
``molpallete_preprocess/enumerate_rgroups.py``; this module loads it and serves
its graphs in batches so the model can embed them into the retrieval
co-embedding space.

The vocabulary is *static* — it depends only on the corpus.  The embedded library
is not: it depends on the R-group projector's current weights and must be rebuilt
whenever those change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch

from .molpla_prep_bridge import load_vocabulary

__all__ = ["RGroupLibraryVocab"]


class RGroupLibraryVocab:
    """Loaded R-group vocabulary, ordered by descending frequency.

    Row order is frequency-descending and fixed at construction, so a FAISS row id
    means the same R-group for the life of the object and can be cached alongside
    an index.

    Attributes
    ----------
    hashes
        Row order: WL subgraph hash per row.  This is the retrieval key, and it is
        the same key the corpus stores per R-group and the contrastive loss uses
        for multi-positive grouping.
    smiles
        Human-readable label per row.  **Not** unique: two structurally different
        masked graphs can strip to the same SMILES, which is exactly why the hash
        is the key.
    counts
        Corpus occurrences per row.
    """

    def __init__(
        self,
        path: str | Path,
        max_entries: Optional[int] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"no R-group vocabulary at {self.path}. Build it with:\n"
                f"  python enumerate_rgroups.py --corpus <corpus> "
                f"--output {self.path}"
            )
        vocab = load_vocabulary(self.path)
        self.provenance = dict(vocab.provenance)

        # The corpus's stored rgroup_hashes, this library's keys and the loss's
        # multi-positive grouping must all come from the same hash definition.
        # A mismatch degrades retrieval silently -- every query misses because
        # its target key does not exist in the library -- so fail loudly.
        vocab_hv = self.provenance.get("graph_hash_version")
        corpus_hv = self.provenance.get("corpus_graph_hash_version")
        if vocab_hv is not None and corpus_hv is not None and vocab_hv != corpus_hv:
            raise ValueError(
                f"graph hash version mismatch: vocabulary built with v{vocab_hv}, "
                f"corpus with v{corpus_hv}. Rebuild whichever is stale."
            )

        order = vocab.keys_by_frequency()
        if max_entries is not None:
            order = order[:max_entries]
            self.provenance["truncated_to"] = max_entries

        self.hashes: List[str] = list(order)
        self.smiles: List[str] = [vocab.entries[h].smiles for h in order]
        self.counts: np.ndarray = np.array(
            [vocab.entries[h].count for h in order], dtype=np.int64
        )
        self._graphs = [vocab.entries[h].graph for h in order]
        self._row_of_hash = {h: i for i, h in enumerate(self.hashes)}
        self.device = device

    def __len__(self) -> int:
        return len(self.hashes)

    def row_of(self, key: str) -> int:
        """FAISS row of *key*, or ``-1`` if it is not in the library."""
        return self._row_of_hash.get(key, -1)

    def rows_of(self, keys: Sequence[str]) -> np.ndarray:
        return np.array([self._row_of_hash.get(k, -1) for k in keys], dtype=np.int64)

    def batches(self, batch_size: int = 1024) -> Iterator[Batch]:
        """Yield the library's graphs as PyG batches, in row order."""
        for start in range(0, len(self._graphs), batch_size):
            yield Batch.from_data_list(self._graphs[start : start + batch_size])

    @property
    def frequency_prior(self) -> np.ndarray:
        """``p(row)`` over the library — the reference for any Hit@K claim."""
        total = self.counts.sum()
        return self.counts / total if total else np.zeros_like(self.counts, dtype=float)

    def effective_size(self) -> float:
        """``exp(H)`` of the frequency distribution: how many-way retrieval *is*.

        A 60,000-row library where one R-group is 28% of occurrences is not a
        60,000-way problem.  Report this next to any Hit@K.
        """
        probs = self.frequency_prior
        nonzero = probs[probs > 0]
        if nonzero.size == 0:
            return 0.0
        return float(np.exp(-(nonzero * np.log(nonzero)).sum()))

    def prior_hit_at_k(self, target_rows: np.ndarray, k: int) -> float:
        """Hit@K of the constant "always return the K most frequent" predictor.

        Rows are frequency-ordered, so that predictor returns rows ``0..K-1``.
        This is the baseline a trained retriever has to beat.  MolPLA reported a
        `Popularity Choice` baseline for the same reason — and on DrugBank it beat
        MolPLA on MRR.
        """
        valid = target_rows[target_rows >= 0]
        if valid.size == 0:
            return 0.0
        return float((valid < k).mean())
