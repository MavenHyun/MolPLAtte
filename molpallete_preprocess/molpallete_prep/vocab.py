"""Vocabulary file format for masked-linker sub-graphs.

A *vocabulary* is a dict keyed by ``subgraph_hash`` (a Weisfeiler-Lehman
graph-isomorphism hash; see :func:`molpallete_prep.anchored.graph_ops.subgraph_hash`)
mapping to a per-entry record::

    {
        "count":      int,            # number of occurrences across the dataset
        "graph":      MolPalleteData,     # canonical example, linker_metas stripped
        "num_atoms":  int,
        "num_masked": int,            # number of is_linker=True atoms
    }

Why strip ``linker_metas``? Because two graphs that share the same hash
have identical post-mask state by construction, but their *pre-mask*
features (atomic_num at the masked atom, etc.) generally differ across
instances. Linker metas are per-instance bookkeeping, not part of the
vocabulary identity — keeping them would bloat the file and might confuse
downstream code into thinking the vocab entry encodes one specific parent.

I/O is via plain ``pickle`` + ``zlib`` so the file is portable to any
Python env that can import ``molpallete_prep``.

Public API:
  - ``build_vocab(items)`` — list[MolPalleteData] -> vocab dict (with counts).
  - ``merge_vocab(*vocabs)`` — combine N vocab dicts (sum counts, keep
    first-seen graph as the canonical example).
  - ``save_vocab(vocab, path)`` / ``load_vocab(path)``.
  - ``strip_linker_metas(data, in_place=False)`` — utility used by
    ``build_vocab``; exposed for callers who want to strip standalone.
"""
from __future__ import annotations

import pickle
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Union

import torch
from torch_geometric.data import Data

from .graph_hash import subgraph_hash
from .mol_features import MolPalleteData


def strip_linker_metas(data: Data, in_place: bool = False) -> Data:
    """Return ``data`` with ``linker_metas`` cleared. Useful when storing
    canonical examples in a vocab (where pre-mask features aren't part of
    the identity)."""
    out = data if in_place else data.clone()
    if hasattr(out, "linker_metas"):
        out.linker_metas = {}
    if hasattr(out, "linker_meta"):
        try:
            delattr(out, "linker_meta")
        except AttributeError:
            pass
    return out


def _entry_for(graph: Data, count: int = 1) -> dict:
    """Build a single vocab entry from a (canonical) graph + occurrence count."""
    g = strip_linker_metas(graph)
    return {
        "count":      int(count),
        "graph":      g,
        "num_atoms":  int(g.num_nodes),
        "num_masked": int(g.is_linker.sum().item())
                      if hasattr(g, "is_linker") else 0,
    }


def build_vocab(items: Iterable[Data],
                hash_kwargs: dict = None,
                ) -> Dict[str, dict]:
    """Build a vocabulary from an iterable of ``MolPalleteData`` objects.

    Items can repeat — duplicates increment the corresponding entry's
    ``count``. The first occurrence per hash is kept as the canonical
    ``graph``.

    Parameters
    ----------
    items : iterable of Data
    hash_kwargs : dict, optional
        Forwarded to :func:`subgraph_hash` (``iterations``, ``digest_size``).
    """
    hk = hash_kwargs or {}
    vocab: Dict[str, dict] = {}
    for g in items:
        h = subgraph_hash(g, **hk)
        if h in vocab:
            vocab[h]["count"] += 1
        else:
            vocab[h] = _entry_for(g, count=1)
    return vocab


def merge_vocab(*vocabs: Dict[str, dict]) -> Dict[str, dict]:
    """Sum counts across multiple vocab dicts; keep the first-seen
    canonical graph per hash. Useful when combining per-shard or
    per-worker results in a multiprocessing pipeline."""
    out: Dict[str, dict] = {}
    for v in vocabs:
        for h, entry in v.items():
            if h in out:
                out[h]["count"] += entry["count"]
            else:
                # shallow copy is fine — entry's graph is immutable from here on
                out[h] = dict(entry)
    return out


def save_vocab(vocab: Dict[str, dict],
               path: Union[str, Path],
               compress: bool = True) -> None:
    """Pickle the vocab dict to ``path``. With ``compress=True`` (default)
    the bytes are zlib-compressed and the suffix ``.gz`` is recommended."""
    blob = pickle.dumps(vocab, protocol=pickle.HIGHEST_PROTOCOL)
    if compress:
        blob = zlib.compress(blob, 6)
    Path(path).write_bytes(blob)


def load_vocab(path: Union[str, Path]) -> Dict[str, dict]:
    """Inverse of :func:`save_vocab`. Auto-detects zlib compression by
    looking at the magic bytes (``\\x78`` is the zlib header)."""
    blob = Path(path).read_bytes()
    if blob[:1] == b"\x78":
        blob = zlib.decompress(blob)
    return pickle.loads(blob)


def vocab_summary(vocab: Dict[str, dict], top_k: int = 10) -> str:
    """Human-readable one-screen summary string."""
    if not vocab:
        return "vocab: empty"
    sizes = [e["num_atoms"] for e in vocab.values()]
    masked = [e["num_masked"] for e in vocab.values()]
    counts = [e["count"] for e in vocab.values()]
    total_inst = sum(counts)
    lines = [
        f"vocab entries  : {len(vocab):>8d}",
        f"total instances: {total_inst:>8d}",
        f"atoms  min/median/max: {min(sizes)}/{sorted(sizes)[len(sizes)//2]}/{max(sizes)}",
        f"masks  min/median/max: {min(masked)}/{sorted(masked)[len(masked)//2]}/{max(masked)}",
        f"top {top_k} by count:",
    ]
    for h, e in sorted(vocab.items(), key=lambda kv: -kv[1]["count"])[:top_k]:
        lines.append(f"  {h[:16]}...  count={e['count']:>5d}  atoms={e['num_atoms']:>2d}  masked={e['num_masked']}")
    return "\n".join(lines)
