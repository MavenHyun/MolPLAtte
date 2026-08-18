#!/usr/bin/env python
"""Build the R-group library vocabulary from a MolPallete corpus.

MolPLA's R-Group Retrieval task retrieves against a library of every recommendable
R-group in the corpus, not against in-batch negatives.  This script builds the
static half of that library: the distinct R-groups, their canonical masked graphs,
their occurrence counts and their condition vectors.  The model-dependent half —
the embedded FAISS index — is built in the training repo from a checkpoint, and is
rebuilt whenever the projector changes.

Which R-groups are recommendable
--------------------------------
Every R-group of every stored decomposition.  MolPLA restricted the library to
R-groups that appear *detached* in some instance, but in MolPallete every R-group
of a decomposition is detached under some ``islinked`` pattern (the subset space is
all of ``2^k - 1``), so the two definitions coincide.

Filtering
---------
MolPLA additionally dropped instances whose R-groups were mostly "common" ones
(above the 99.99th percentile of occurrence), because retrieval scores are
dominated by a handful of trivial groups like ``-OH``.  ``--drop-top-percentile``
reproduces that filter on the library side; it is **off by default** because the
right threshold depends on the corpus, and because silently shrinking the
retrieval space inflates every metric computed against it.  Whatever is dropped is
logged and recorded in the vocabulary provenance.

Example
-------
::

    python enumerate_rgroups.py \\
      --corpus /home/mogan/corpora/molpallete/flavor_v1/macfrag \\
      --output /home/mogan/corpora/molpallete/flavor_v1/macfrag/rgroup_vocab.pkl.gz \\
      --workers 88
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from molpallete_prep import __version__
from molpallete_prep.graph_hash import subgraph_hash
from molpallete_prep.graph_ops import detach_rgroups_multi
from molpallete_prep.lmdb_store import hydrate
from molpallete_prep.mol_features import pyg_to_mol
from molpallete_prep.preprocess import atomic_write_json, path_for
from molpallete_prep.rgroup_library import (
    RGroupVocabulary,
    rgroup_smiles,
    save_vocabulary,
)

_W: Dict[str, object] = {}


def _init_worker(config: dict) -> None:
    _W.clear()
    _W.update(config)


def _scan_one(mol_id: str) -> Optional[RGroupVocabulary]:
    """Materialise every R-group of one corpus record into a mini-vocabulary."""
    try:
        raw = torch.load(
            str(path_for(Path(_W["corpus"]), mol_id, _W["layout"])), weights_only=False
        )
    except (FileNotFoundError, EOFError, RuntimeError, OSError):
        return None

    record = hydrate(raw)
    original = record["original"]
    # Rebuild the molecule FROM THE GRAPH, not by re-parsing record["smiles"].
    # The stored R-group atom indices refer to the graph's atom ordering, and
    # RDKit re-orders atoms when it parses a canonical SMILES -- so re-parsing
    # would silently index the wrong atoms and emit disconnected nonsense
    # ("C.*O", "*Cccccc"). pyg_to_mol preserves the ordering by construction.
    try:
        mol = pyg_to_mol(original, sanitize=True)
    except Exception:
        mol = None

    vocab = RGroupVocabulary()
    for decomp in record["decompositions"]:
        stored_hashes = decomp.get("rgroup_hashes") or []
        condvecs = decomp.get("rgroup_condvecs")
        for index, rgroup in enumerate(decomp["rgroups"]):
            atoms = tuple(int(a) for a in rgroup["rgroup_atoms"])
            core_linker = int(rgroup["core_linker"])
            rgroup_linker = int(rgroup["rgroup_linker"])
            try:
                _template, detached = detach_rgroups_multi(
                    original, [(atoms, core_linker, rgroup_linker)], store_orig=False
                )
            except Exception:
                continue
            graph = detached[0]

            key = (
                stored_hashes[index]
                if index < len(stored_hashes) and stored_hashes[index]
                else subgraph_hash(graph)
            )
            if not key:
                continue

            condvec = None
            if condvecs is not None and index < len(condvecs):
                condvec = np.asarray(condvecs[index], dtype=np.uint8)

            smiles = (
                rgroup_smiles(mol, atoms, rgroup_linker) if mol is not None else ""
            )
            vocab.add(
                key,
                graph=graph,
                smiles=smiles,
                condvec=condvec,
                n_atoms=int(graph.num_nodes),
            )
    return vocab


def _scan_batch(mol_ids: List[str]) -> RGroupVocabulary:
    merged = RGroupVocabulary()
    for mol_id in mol_ids:
        partial = _scan_one(mol_id)
        if partial is not None:
            merged.merge(partial)
    return merged


def _chunked(items: List[str], size: int) -> Iterator[List[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", required=True, help="a built corpus directory")
    parser.add_argument(
        "--output",
        default=None,
        help="defaults to <corpus>/rgroup_vocab.pkl.gz",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--limit-mols", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25000)
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="drop R-groups occurring fewer than this many times",
    )
    parser.add_argument(
        "--drop-top-percentile",
        type=float,
        default=None,
        help="drop R-groups above this occurrence percentile (MolPLA used 99.99 "
        "to remove trivial groups like -OH). Off by default: shrinking the "
        "retrieval space inflates every metric measured against it.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    corpus = Path(args.corpus)
    meta_path = corpus / "__meta__.json"
    if not meta_path.is_file():
        print(f"ERROR: no __meta__.json at {corpus}", file=sys.stderr)
        return 1
    corpus_meta = json.loads(meta_path.read_text())
    layout = corpus_meta.get("layout", "hash3")
    ids = list(corpus_meta["ids"])
    if args.limit_mols is not None:
        ids = ids[: args.limit_mols]

    output = Path(args.output) if args.output else corpus / "rgroup_vocab.pkl.gz"

    for key, value in [
        ("corpus", str(corpus)),
        ("method", corpus_meta.get("method")),
        ("records", f"{len(ids):,}"),
        ("layout", layout),
        ("workers", args.workers),
        ("output", str(output)),
    ]:
        print(f"[info] {key:12s} : {value}", flush=True)

    vocab = RGroupVocabulary()
    started = time.time()
    processed = 0
    last_report = 0

    with mp.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=({"corpus": str(corpus), "layout": layout},),
    ) as pool:
        for partial in pool.imap_unordered(
            _scan_batch, _chunked(ids, args.batch_size), chunksize=1
        ):
            vocab.merge(partial)
            processed += args.batch_size
            if processed - last_report >= args.progress_every:
                elapsed = time.time() - started
                print(
                    f"[prog] {min(processed, len(ids)):>9,}/{len(ids):,} records  "
                    f"{elapsed:7.1f}s  {min(processed, len(ids)) / max(elapsed, 1e-9):7.1f} rec/s  "
                    f"vocab={len(vocab):,}",
                    flush=True,
                )
                last_report = processed

    n_before = len(vocab)
    dropped_rare = dropped_common = 0

    if args.min_count > 1:
        keep = {k: e for k, e in vocab.entries.items() if e.count >= args.min_count}
        dropped_rare = n_before - len(keep)
        vocab.entries = keep

    if args.drop_top_percentile is not None:
        counts = np.array([e.count for e in vocab.entries.values()])
        threshold = np.percentile(counts, args.drop_top_percentile)
        keep = {k: e for k, e in vocab.entries.items() if e.count <= threshold}
        dropped_common = len(vocab.entries) - len(keep)
        vocab.entries = keep
        print(
            f"[filter] dropped {dropped_common:,} R-groups above the "
            f"{args.drop_top_percentile} percentile (count > {threshold:.0f})",
            flush=True,
        )

    elapsed = time.time() - started
    vocab.provenance = {
        "molpallete_prep_version": __version__,
        "corpus": str(corpus),
        "corpus_method": corpus_meta.get("method"),
        "corpus_source": corpus_meta.get("source"),
        "corpus_n_records": corpus_meta.get("n_records"),
        "n_records_scanned": len(ids),
        "condvec_mode": corpus_meta.get("condvec_mode"),
        "condvec_dim": corpus_meta.get("condvec_dim"),
        "rdkit_version": corpus_meta.get("rdkit_version"),
        "min_count": args.min_count,
        "drop_top_percentile": args.drop_top_percentile,
        "n_distinct_before_filter": n_before,
        "n_dropped_rare": dropped_rare,
        "n_dropped_common": dropped_common,
        "n_distinct": len(vocab),
        "effective_size": vocab.effective_size(),
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    save_vocabulary(vocab, output)
    atomic_write_json(output.with_suffix(".meta.json"), vocab.provenance)

    print(f"\n{vocab.summary(top_k=15)}")
    print(
        f"\n[done] scanned {len(ids):,} records in {elapsed:.1f}s\n"
        f"[done] vocabulary : {len(vocab):,} distinct R-groups "
        f"(effective {vocab.effective_size():,.0f})\n"
        f"[done] written    : {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
