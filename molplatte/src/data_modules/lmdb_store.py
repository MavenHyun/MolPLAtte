"""LMDB-backed store for preprocessed MolPLAtte corpus.

Layout
------

One LMDB environment per (paradigm, method). Concrete keys:

* ``b"__meta__"``       — JSON metadata + flat per-record counts
* ``b"r:00000000"`` …    — one zlib(pickle(payload)) entry per molecule

The metadata key holds everything a sampler needs to build its flat
``(mol_idx, decomp_idx)`` / ``(mol_idx, partition_idx)`` index without
touching any record payload:

.. code-block:: json

    {
        "schema_version": 1,
        "paradigm":       "anchored" | "fragments",
        "method":         "naveja_recap" | "brics" | ...,
        "n_records":      83908,
        "ids":            ["ZINC...", ...],         // len == n_records
        "n_decomps":      [9, 7, 12, 0, ...],        // anchored only
        "n_partitions":   [1, 1, 1, ...],            // fragments only
        "n_fragments_per_partition": [[k_0_0, k_0_1, ...], ...],
        "extra":          {}                         // free-form
    }

Reader
------

:class:`LMDBStore` opens the environment read-only, mmap-shared across
DataLoader workers. ``__getitem__(idx)`` does one transaction, fetches
the value, decompresses + unpickles. ~50 µs of LMDB overhead per call
plus the inherent decompress/unpickle cost (~17 ms).

Writer
------

:class:`LMDBWriter` is a buffered writer used by the conversion script
or by future preprocessing. Buffers ``put()`` calls and flushes on
:meth:`commit` or :meth:`close` (context-managed).
"""
from __future__ import annotations

import json
import pickle
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import lmdb
import torch
import zstandard as zstd

try:
    import lz4.frame as _lz4_frame
    _LZ4_OK = True
except ImportError:
    _LZ4_OK = False


SCHEMA_VERSION = 2
"""Bumped from 1 → 2 with the switch from zlib → zstd compression and
the addition of a ``compression`` field in ``__meta__``. Old (v1)
stores have no ``compression`` key and are assumed zlib for backward
compatibility on the read path."""

# zstd compression level. 3 hits the sweet spot of ratio vs speed for
# our small (~5-10 KB) record blobs.
ZSTD_LEVEL = 3

# ~16 GiB upper bound by default. LMDB allocates virtual address space
# this large but only uses what's actually written. Override per call
# if you expect >16 GiB.
DEFAULT_MAP_SIZE = 16 * 1024 * 1024 * 1024


def _record_key(idx: int) -> bytes:
    """Key format for record idx: ``r:NNNNNNNN`` (zero-padded for sortability)."""
    return f"r:{idx:08d}".encode()


# ---------------------------------------------------------------------------
# portable (de)hydration — keep ``molplatte.*`` class qualnames out of the
# pickle bytes so the LMDB records can be loaded anywhere torch +
# torch_geometric are installed, with no molplatte package required.
# ---------------------------------------------------------------------------

_HELPERS_CACHE = None


def _import_helpers():
    """Resolve the (de)hydration helpers from whichever layout this
    module is sitting in:

    * source-dev/molplatte/data_modules/  → data_types.py is a sibling
    * MolPLAtte_prep/molplatte/              → data_types lives in fragments/

    Cached after first call — dehydrate/hydrate recurse per key, so a
    fresh import lookup per call cost tens of ms per record.
    """
    global _HELPERS_CACHE
    if _HELPERS_CACHE is not None:
        return _HELPERS_CACHE
    from .mol_features import (
        MolPLAtteData, data_to_portable, portable_to_data, is_portable_data,
    )
    try:
        from .data_types import (
            FragmentPartition,
            fragment_partition_to_portable,
            portable_to_fragment_partition,
            is_portable_partition,
        )
    except ImportError:
        from .fragments.data_types import (
            FragmentPartition,
            fragment_partition_to_portable,
            portable_to_fragment_partition,
            is_portable_partition,
        )
    _HELPERS_CACHE = (MolPLAtteData, data_to_portable, portable_to_data, is_portable_data,
                      FragmentPartition, fragment_partition_to_portable,
                      portable_to_fragment_partition, is_portable_partition)
    return _HELPERS_CACHE


import numpy as _np


def dehydrate(obj):
    """Recursively replace MolPLAtteData / FragmentPartition with portable
    dicts and convert bare ``torch.Tensor`` values to ``numpy.ndarray``.

    Storing numpy avoids the ``torch.load``/``_load_from_bytes`` codepath
    at unpickle time, which is not fork-safe under PyTorch DataLoader
    workers.
    """
    (MolPLAtteData, data_to_portable, _ptd, _ipd,
     FragmentPartition, fragment_partition_to_portable, _ptfp, _ipfp) = _import_helpers()
    if isinstance(obj, MolPLAtteData):
        return data_to_portable(obj)
    if isinstance(obj, FragmentPartition):
        return fragment_partition_to_portable(obj)
    if isinstance(obj, dict):
        return {k: dehydrate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dehydrate(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(dehydrate(x) for x in obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    return obj


def hydrate(obj):
    """Inverse of :func:`dehydrate`. Reconstructs MolPLAtteData /
    FragmentPartition from portable dicts. Bare ``numpy.ndarray`` values
    (from v2 records) are converted back to ``torch.Tensor``.
    """
    (_MD, _dtp, portable_to_data, is_portable_data,
     _FP, _fptp, portable_to_fragment_partition, is_portable_partition) = _import_helpers()
    if isinstance(obj, dict):
        if is_portable_data(obj):
            return portable_to_data(obj)
        if is_portable_partition(obj):
            return portable_to_fragment_partition(obj)
        return {k: hydrate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [hydrate(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(hydrate(x) for x in obj)
    if isinstance(obj, _np.ndarray):
        return torch.from_numpy(obj)
    return obj


import os as _os

_ENV_CACHE: Dict[str, "lmdb.Environment"] = {}
_ENV_CACHE_PID: int = _os.getpid()


def _reset_env_cache() -> None:
    global _ENV_CACHE, _ENV_CACHE_PID
    _ENV_CACHE = {}
    _ENV_CACHE_PID = _os.getpid()


def _shared_env(path: Path, *, readahead: bool, max_readers: int) -> lmdb.Environment:
    global _ENV_CACHE_PID
    pid = _os.getpid()
    if pid != _ENV_CACHE_PID:
        _reset_env_cache()
    key = str(path.resolve())
    env = _ENV_CACHE.get(key)
    if env is None:
        env = lmdb.open(
            str(path),
            readonly=True,
            lock=False,
            readahead=readahead,
            meminit=False,
            max_readers=max_readers,
            subdir=True,
        )
        _ENV_CACHE[key] = env
    return env


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------

class LMDBStore:
    """Read-only mmap-shared view over an LMDB of preprocessed molecules.

    Safe to construct in multiple :class:`torch.utils.data.DataLoader`
    workers: each worker calls :meth:`__init__` independently, and the
    underlying file is mmaped so workers share OS page cache.

    For PyTorch's ``num_workers > 0`` setup, do NOT open the LMDB in
    the main process and pass the handle to workers — instead, lazily
    open in each worker (the dataset class below does this).
    """

    def __init__(self, path: str | Path,
                 readahead: bool = False,
                 max_readers: int = 256):
        self.path = Path(path)
        self._readahead = readahead
        self._max_readers = max_readers
        env = self._ensure_env()
        with env.begin() as txn:
            blob = txn.get(b"__meta__")
            if blob is None:
                raise RuntimeError(
                    f"LMDB at {self.path} has no '__meta__' key — not a "
                    f"MolPLAtte store, or never written")
            self.meta: Dict[str, Any] = json.loads(blob.decode())
        # Compression dispatch — v2+ records carry the field; v1 stores
        # are zlib by convention.
        self.compression = self.meta.get("compression", "zlib")
        self._zstd_dec = zstd.ZstdDecompressor() if self.compression == "zstd" else None
        self._zstd_pid = _os.getpid()

    # --- env handling: process-shared cache (LMDB allows only one
    # Environment object per path per process) --------------------------

    def _ensure_env(self) -> lmdb.Environment:
        return _shared_env(self.path,
                           readahead=self._readahead,
                           max_readers=self._max_readers)

    # PyTorch's DataLoader can spawn workers via fork OR spawn. The env
    # cache is per-process so each worker re-opens the env on first
    # __getitem__. We DON'T pickle the lmdb handle.
    def __getstate__(self):
        return {"path": self.path,
                "_readahead": self._readahead,
                "_max_readers": self._max_readers,
                "meta": self.meta,
                "compression": self.compression}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._zstd_dec = (zstd.ZstdDecompressor()
                          if self.compression == "zstd" else None)
        self._zstd_pid = _os.getpid()

    def close(self):
        """Close the shared env (affects all LMDBStore instances against
        the same path in this process). Rarely needed."""
        key = str(self.path.resolve())
        env = _ENV_CACHE.pop(key, None)
        if env is not None:
            env.close()

    # --- accessors ----------------------------------------------------

    def __len__(self) -> int:
        return int(self.meta["n_records"])

    @property
    def paradigm(self) -> str:
        return self.meta["paradigm"]

    @property
    def method(self) -> str:
        return self.meta.get("method", "?")

    @property
    def ids(self) -> List[str]:
        return self.meta["ids"]

    def _decompress(self, blob: bytes) -> bytes:
        if self.compression == "zstd":
            pid = _os.getpid()
            if pid != self._zstd_pid:
                self._zstd_dec = zstd.ZstdDecompressor()
                self._zstd_pid = pid
            return self._zstd_dec.decompress(blob)
        if self.compression == "lz4":
            if not _LZ4_OK:
                raise RuntimeError("compression=lz4 but python-lz4 not installed; pip install lz4")
            return _lz4_frame.decompress(blob)
        if self.compression == "zlib":
            return zlib.decompress(blob)
        if self.compression == "none":
            return blob
        raise ValueError(f"unknown compression in __meta__: {self.compression!r}")

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        env = self._ensure_env()
        with env.begin(buffers=True) as txn:
            blob = txn.get(_record_key(idx))
            if blob is None:
                raise KeyError(f"missing key r:{idx:08d}")
            payload = pickle.loads(self._decompress(bytes(blob)))
            return hydrate(payload)


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------

class LMDBWriter:
    """Buffered writer for an LMDB store.

    Use as a context manager. ``put_record`` adds one molecule's
    payload; the writer batches them by ``commit_every`` records into
    LMDB write transactions for throughput.

    Example::

        with LMDBWriter("path.lmdb", paradigm="anchored", method="naveja_recap") as w:
            for payload in payloads:
                w.put_record(payload)
        # __meta__ is written at close() with the accumulated index.
    """

    def __init__(self,
                 path: str | Path,
                 paradigm: str,
                 method: str,
                 map_size: int = DEFAULT_MAP_SIZE,
                 commit_every: int = 1000,
                 overwrite: bool = False,
                 compression: str = "zstd"):
        self.path = Path(path)
        self.paradigm = paradigm
        self.method = method
        self.commit_every = commit_every
        self.compression = compression
        if compression not in ("zstd", "lz4", "zlib", "none"):
            raise ValueError(f"compression must be zstd/lz4/zlib/none, got {compression!r}")
        if compression == "lz4" and not _LZ4_OK:
            raise RuntimeError("compression=lz4 but python-lz4 not installed; pip install lz4")
        self._zstd_enc = (zstd.ZstdCompressor(level=ZSTD_LEVEL)
                          if compression == "zstd" else None)
        if self.path.exists() and overwrite:
            import shutil
            shutil.rmtree(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(
            str(self.path),
            map_size=map_size,
            subdir=True,
            map_async=True,
            writemap=True,
            meminit=False,
            sync=False,
        )
        self._txn: Optional[lmdb.Transaction] = None
        self._pending = 0
        self._n_records = 0
        self._ids: List[str] = []
        self._n_decomps: List[int] = []
        self._n_partitions: List[int] = []
        self._n_frags_per_p: List[List[int]] = []

    # --- transaction handling ----------------------------------------

    def _begin(self):
        if self._txn is None:
            self._txn = self.env.begin(write=True)

    def commit(self):
        if self._txn is not None:
            self._txn.commit()
            self._txn = None
            self._pending = 0

    # --- public API ---------------------------------------------------

    def _compress(self, raw: bytes) -> bytes:
        if self.compression == "zstd":
            return self._zstd_enc.compress(raw)
        if self.compression == "lz4":
            return _lz4_frame.compress(raw, compression_level=1)
        if self.compression == "zlib":
            return zlib.compress(raw, 3)
        return raw

    def put_record(self, payload: dict, blob: Optional[bytes] = None):
        """Add one molecule's preprocessed payload.

        Pass ``blob`` (already-compressed pickle bytes) if you produced
        it elsewhere — it MUST match this writer's compression mode.
        Otherwise we ``dehydrate`` (drop ``molplatte.*`` class refs),
        pickle, and compress ``payload`` here using ``self.compression``.
        """
        if blob is None:
            portable = dehydrate(payload)
            blob = self._compress(
                pickle.dumps(portable, protocol=pickle.HIGHEST_PROTOCOL))
        meta = {
            "mol_id":      payload["mol_id"],
            "n_decomps":    int(payload.get("n_decomps", 0)),
            "n_partitions": int(payload.get("n_partitions", 0)),
            "n_fragments_per_partition": (
                [int(P.get("n_fragments", 0)) for P in payload["partitions"]]
                if "partitions" in payload else []),
        }
        self.put_blob_meta(blob, meta)

    def put_blob_meta(self, blob: bytes, meta: Dict[str, Any]):
        """Add one record from a pre-encoded blob + a tiny metadata dict.

        This is the cheap path for multiprocessing workers: they already
        pickled + compressed the payload, so the parent shouldn't have
        to ship the full payload back across the mp boundary just to
        read 3 integers.

        ``meta`` keys: ``mol_id``, ``n_decomps``, ``n_partitions``,
        ``n_fragments_per_partition`` (list of ints).
        """
        self._begin()
        idx = self._n_records
        self._txn.put(_record_key(idx), blob)
        self._n_records += 1
        self._pending += 1
        self._ids.append(meta["mol_id"])
        self._n_decomps.append(int(meta.get("n_decomps", 0)))
        self._n_partitions.append(int(meta.get("n_partitions", 0)))
        self._n_frags_per_p.append(list(meta.get("n_fragments_per_partition", [])))
        if self._pending >= self.commit_every:
            self.commit()

    def write_meta(self, extra: Optional[Dict[str, Any]] = None):
        """Write the ``__meta__`` key with the accumulated index."""
        meta = {
            "schema_version":            SCHEMA_VERSION,
            "paradigm":                  self.paradigm,
            "method":                    self.method,
            "compression":               self.compression,
            "n_records":                 self._n_records,
            "ids":                       self._ids,
            "n_decomps":                 self._n_decomps,
            "n_partitions":              self._n_partitions,
            "n_fragments_per_partition": self._n_frags_per_p,
            "extra":                     extra or {},
        }
        self._begin()
        self._txn.put(b"__meta__", json.dumps(meta).encode())
        self.commit()

    def close(self):
        if self._n_records > 0:
            self.write_meta()
        self.commit()
        self.env.sync()
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            # Don't write __meta__ on exception — leaves the store usable
            # for resume but flagged as incomplete by the absence of meta.
            self.commit()
            self.env.sync()
            self.env.close()


__all__ = ["LMDBStore", "LMDBWriter", "SCHEMA_VERSION"]
