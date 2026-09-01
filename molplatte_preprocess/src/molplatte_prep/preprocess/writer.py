"""On-disk corpus layout for MolPLAtte.

One ``.pt`` per molecule under a sharded directory tree, plus two JSON sidecars:

``__meta__.json``
    What a *consumer* needs: the parallel per-record index arrays (``ids``,
    ``n_decomps``, ``n_rgroups_per_decomp``) so a Dataset can build its sampler
    index without opening a single record, plus the full provenance block.  The
    provenance is deliberately stored **inside** the corpus: MolDAM_prep lost the
    run manifest for seven corpora during a directory move and had to reconstruct
    decomposition ratios from file mtimes.

``__manifest__.json``
    What an *auditor* needs: resolved inputs, worker settings, status counts,
    throughput, wall-clock.

Differences from the MolDAM_prep original this was adapted from:

* **Metadata writes are atomic** (temp file + :func:`os.replace`).  ``__meta__.json``
  runs to tens of MB on a large corpus and a torn write destroys the index; the
  MolDAM_prep handoff flagged this as its clearest unfixed defect.
* **A ``hash3`` layout exists.**  Suffix-slicing a molecule id assumes fixed-width,
  uniformly-distributed ids.  That holds for 16-digit ZINC ids; it does not hold
  for ``FDB4`` / ``FDB25595`` / ``CNP0252853``, which bucket very unevenly.
  ``hash3`` buckets on a blake2b digest instead, which is uniform by construction.
* **Records are resumable**: :func:`existing_ids` lets a re-run skip molecules
  already on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import torch

from ..lmdb_store import dehydrate

__all__ = [
    "SHARD_LAYOUTS",
    "path_for",
    "atomic_write_json",
    "write_record",
    "existing_ids",
    "write_meta",
    "write_manifest",
]

#: ``layout -> number of characters taken from the id suffix``.  ``hash3`` is
#: special-cased: it buckets on a digest rather than on the id itself.
SHARD_LAYOUTS: Dict[str, int] = {
    "flat": 0,
    "shard2": 2,
    "shard3": 3,
    "hash3": -1,
}


def _bucket_of(mol_id: str, layout: str) -> str:
    """Directory bucket for *mol_id* under *layout* (``""`` means no bucket)."""
    if layout == "flat":
        return ""
    if layout == "hash3":
        digest = hashlib.blake2b(mol_id.encode("utf-8"), digest_size=2).hexdigest()
        return digest[:3]
    n = SHARD_LAYOUTS.get(layout, 0)
    if n <= 0:
        return ""
    return mol_id[-n:]


def path_for(root: Path, mol_id: str, layout: str) -> Path:
    """Absolute path of the record for *mol_id*."""
    if layout not in SHARD_LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; available: {list(SHARD_LAYOUTS)}")
    bucket = _bucket_of(mol_id, layout)
    return (root / bucket / f"{mol_id}.pt") if bucket else (root / f"{mol_id}.pt")


def atomic_write_json(path: Path, payload: dict) -> None:
    """Serialise *payload* to *path* without ever leaving a torn file behind.

    Writes to a temp file in the destination directory (so the rename stays on
    one filesystem), fsyncs, then :func:`os.replace`, which is atomic on POSIX.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_record(root: Path, mol_id: str, payload: dict, layout: str) -> Path:
    """Dehydrate and persist one molecule record.

    ``dehydrate`` converts every PyG ``Data`` to the portable numpy form, so the
    pickle carries no ``molplatte_prep`` class qualnames and no torch tensors --
    the latter matters because unpickling a torch tensor is not fork-safe under
    multi-worker DataLoaders.
    """
    out = path_for(Path(root), mol_id, layout)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dehydrate(payload), str(out))
    return out


def existing_ids(root: Path, ids: Iterable[str], layout: str) -> Set[str]:
    """Subset of *ids* whose record file already exists -- the resume predicate."""
    root = Path(root)
    return {i for i in ids if path_for(root, i, layout).is_file()}


def write_meta(
    root: Path,
    *,
    ids: List[str],
    n_decomps: List[int],
    n_rgroups_per_decomp: List[List[int]],
    provenance: dict,
) -> Path:
    """Write the consumer index ``__meta__.json``.

    ``ids``, ``n_decomps`` and ``n_rgroups_per_decomp`` are parallel arrays of
    equal length; the verification recipe in ``docs/corpus_format.md`` asserts
    exactly that.
    """
    if not (len(ids) == len(n_decomps) == len(n_rgroups_per_decomp)):
        raise ValueError(
            "index arrays disagree: "
            f"{len(ids)} ids, {len(n_decomps)} n_decomps, "
            f"{len(n_rgroups_per_decomp)} n_rgroups_per_decomp"
        )
    path = Path(root) / "__meta__.json"
    atomic_write_json(
        path,
        {
            "n_records": len(ids),
            "ids": ids,
            "n_decomps": n_decomps,
            "n_rgroups_per_decomp": n_rgroups_per_decomp,
            **provenance,
        },
    )
    return path


def write_manifest(root: Path, manifest: dict) -> Path:
    """Write the auditor record ``__manifest__.json`` inside the corpus."""
    path = Path(root) / "__manifest__.json"
    atomic_write_json(path, manifest)
    return path
