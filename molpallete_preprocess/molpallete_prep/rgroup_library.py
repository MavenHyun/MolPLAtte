"""The R-group library — MolPLA's retrieval target space.

MolPLA's R-Group Retrieval framework is not scored against in-batch negatives.
It builds a **library of every recommendable R-group in the corpus** (61,279 of
them on GEOM), embeds all of them with the trained R-group projector, and
retrieves the top 1000 by inner product in that co-embedding space.  MRR and
Hit@K are computed against that full library.  Without it there is no lead
optimization task — an in-batch gallery of a few hundred rows measures something
much easier and reports it under the same name.

Two artefacts, built at different times:

**The vocabulary** (this module, built once per corpus at preprocessing time)
    ``hash -> {graph, smiles, count, condvec, n_atoms}``.  Static: it depends only
    on the corpus, not on any model.

**The vector library** (built in the training repo from a checkpoint)
    The vocabulary's graphs pushed through encoder -> pooling -> R-group
    projector, L2-normalised, in a FAISS index.  This one is **model-dependent and
    must be rebuilt whenever the projector changes** — MolPLA rebuilds it at every
    validation epoch, and so does ``callbacks/RGroupLibraryRetrieval.py``.

Keying
------
MolPLA keys the vocabulary on the R-group's masked SMILES string.  MolPallete keys
on the **Weisfeiler-Lehman subgraph hash** instead and carries SMILES as a label,
because a masked linker atom is not a real chemical entity: two R-groups can share
a SMILES while differing in mask state, and RDKit's canonicalisation of a fragment
with a dummy is not stable across the ways that fragment can be written.  The hash
is what the corpus already stores per R-group and what the contrastive loss already
uses to group multi-positives, so keying on it keeps one identity notion
throughout.
"""

from __future__ import annotations

import gzip
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

__all__ = [
    "RGroupEntry",
    "RGroupVocabulary",
    "rgroup_smiles",
    "save_vocabulary",
    "load_vocabulary",
]


@dataclass
class RGroupEntry:
    """One distinct R-group in the library.

    Attributes
    ----------
    hash
        WL subgraph hash — the vocabulary key.
    graph
        A canonical masked R-group graph (the first occurrence seen).  This is
        what gets embedded to build the vector library.
    smiles
        Human-readable label with an explicit ``*`` attachment point, e.g. ``*O``
        or ``*c1ccccc1``.  Not the key; see the module docstring.
    count
        Occurrences across the corpus.  Drives the frequency prior that retrieval
        metrics must be compared against, and the common/rare filters.
    condvec
        Functional-group condition vector, ``uint8``.
    n_atoms
        Heavy atoms including the masked clone.
    """

    hash: str
    graph: object
    smiles: str = ""
    count: int = 0
    condvec: Optional[np.ndarray] = None
    n_atoms: int = 0


@dataclass
class RGroupVocabulary:
    """``hash -> RGroupEntry``, plus corpus-level provenance."""

    entries: Dict[str, RGroupEntry] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def add(
        self,
        key: str,
        graph: object,
        smiles: str = "",
        condvec: Optional[np.ndarray] = None,
        n_atoms: int = 0,
        count: int = 1,
    ) -> None:
        """Record an occurrence; the first graph seen becomes canonical."""
        entry = self.entries.get(key)
        if entry is None:
            self.entries[key] = RGroupEntry(
                hash=key,
                graph=graph,
                smiles=smiles,
                count=count,
                condvec=condvec,
                n_atoms=n_atoms,
            )
        else:
            entry.count += count
            if not entry.smiles and smiles:
                entry.smiles = smiles

    def merge(self, other: "RGroupVocabulary") -> "RGroupVocabulary":
        """Fold *other* into this one (the multiprocessing reducer)."""
        for key, entry in other.entries.items():
            existing = self.entries.get(key)
            if existing is None:
                self.entries[key] = entry
            else:
                existing.count += entry.count
                if not existing.smiles and entry.smiles:
                    existing.smiles = entry.smiles
        return self

    def keys_by_frequency(self) -> List[str]:
        """Hashes ordered most-frequent first — the row order of the library."""
        return sorted(
            self.entries,
            key=lambda k: (-self.entries[k].count, k),
        )

    def frequency_prior(self) -> Dict[str, float]:
        """``hash -> p(hash)``.  The reference any Hit@K must be judged against.

        MolDAM's headline retrieval figure, originally reported as "47x above
        random", re-scored against this prior landed *at* the prior.  Random is
        the wrong reference for a vocabulary this skewed.
        """
        total = sum(e.count for e in self.entries.values()) or 1
        return {k: e.count / total for k, e in self.entries.items()}

    def effective_size(self) -> float:
        """Perplexity of the frequency distribution — the *usable* vocabulary size.

        A library of 60,000 R-groups where one accounts for 28% of occurrences is
        not a 60,000-way retrieval problem.  ``exp(H)`` states how many-way it
        actually is.
        """
        probs = np.array([e.count for e in self.entries.values()], dtype=np.float64)
        if probs.sum() <= 0:
            return 0.0
        probs /= probs.sum()
        nonzero = probs[probs > 0]
        return float(np.exp(-(nonzero * np.log(nonzero)).sum()))

    def summary(self, top_k: int = 10) -> str:
        if not self.entries:
            return "empty vocabulary"
        ordered = self.keys_by_frequency()
        total = sum(e.count for e in self.entries.values())
        lines = [
            f"{len(self.entries):,} distinct R-groups, {total:,} occurrences",
            f"effective size (exp H): {self.effective_size():,.0f}",
            f"top-1 share: {self.entries[ordered[0]].count / total:.2%}",
            "",
            f"{'count':>10s}  {'share':>7s}  smiles",
        ]
        for key in ordered[:top_k]:
            entry = self.entries[key]
            label = entry.smiles or key[:16]
            lines.append(f"{entry.count:>10,}  {entry.count / total:>6.2%}  {label}")
        return "\n".join(lines)


def rgroup_smiles(
    mol: Chem.Mol, rgroup_atoms: Sequence[int], rgroup_linker: int
) -> str:
    """SMILES for an R-group with an explicit ``*`` attachment point.

    Reproduces MolPLA's label style (``*O``, ``*c1ccccc1``): a dummy atom is bonded
    to the R-group's linker atom so the attachment position is visible, then the
    fragment is canonicalised.  Returns ``""`` on any failure — a label is not
    worth aborting a vocabulary build for.
    """
    try:
        editable = Chem.RWMol(mol)
        dummy = editable.AddAtom(Chem.Atom(0))
        editable.AddBond(dummy, int(rgroup_linker), Chem.BondType.SINGLE)
        atoms = list(rgroup_atoms) + [dummy]
        return Chem.MolFragmentToSmiles(
            editable.GetMol(), atomsToUse=atoms, canonical=True
        )
    except Exception:
        return ""


def save_vocabulary(vocab: RGroupVocabulary, path: str | Path) -> Path:
    """Write the vocabulary as gzipped pickle, graphs in portable numpy form.

    Portable form keeps ``molpallete_prep`` class qualnames and torch tensors out
    of the pickle, so the training repo can load a vocabulary without importing
    this package and without tripping the fork-safety problem that torch tensors
    in pickles cause under multi-worker DataLoaders.
    """
    from .lmdb_store import dehydrate

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": vocab.provenance,
        "entries": {
            key: {
                "hash": entry.hash,
                "graph": dehydrate(entry.graph),
                "smiles": entry.smiles,
                "count": entry.count,
                "condvec": (
                    np.asarray(entry.condvec) if entry.condvec is not None else None
                ),
                "n_atoms": entry.n_atoms,
            }
            for key, entry in vocab.entries.items()
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path


def load_vocabulary(path: str | Path) -> RGroupVocabulary:
    """Read a vocabulary written by :func:`save_vocabulary`."""
    from .lmdb_store import hydrate

    with gzip.open(Path(path), "rb") as handle:
        payload = pickle.load(handle)

    vocab = RGroupVocabulary(provenance=payload.get("provenance", {}))
    for key, raw in payload["entries"].items():
        vocab.entries[key] = RGroupEntry(
            hash=raw["hash"],
            graph=hydrate(raw["graph"]),
            smiles=raw.get("smiles", ""),
            count=int(raw.get("count", 0)),
            condvec=raw.get("condvec"),
            n_atoms=int(raw.get("n_atoms", 0)),
        )
    return vocab
