"""Corpus reader and ``(decomposition, islinked)`` sampler.

The corpus stores, per molecule, the intact graph plus every decomposition's
*bookkeeping* -- core atom indices and per-R-group atom indices, linkers, WL
hashes and condition vectors -- but **not** the detached graphs.  Materialising
all ``2**k - 1`` islinked subsets on disk would multiply the corpus by ~9x at the
measured 2.69 mean R-groups per decomposition, so the detach happens here, once
per ``__getitem__``.

Sampling unit
-------------
MolPLA's augmentation is two-stage: **one molecule yields many putative cores**,
and **one core with n R-groups yields the subsets** -- for each non-empty subset
*k*, the detached R-groups become the retrieval targets and the *remaining* ones
stay attached to form the query template ``P`` (paper Eq. 3).

``sampling_unit`` chooses how much of that is exposed per epoch:

``"decomposition"`` (default, MolPLA-style)
    ``__len__`` is the number of usable ``(molecule, core)`` pairs. Every core is
    visited once per epoch; the islinked subset is drawn per call. On the naveja
    corpus that is 3,372,588 instances per epoch against 411,456 molecules --
    **8.2x more**, and it removes the bias whereby a molecule with one core and a
    molecule with twenty were sampled equally often.

``"molecule"`` (MolDAM-style)
    ``__len__`` is the number of molecules; one core is drawn per call. Cheaper
    per epoch, but a high-core molecule is under-sampled in exactly the cases
    that carry the most structure.

The flat ``(mol, decomp)`` index is built from ``__meta__.json`` alone -- no
record is opened to construct it.
"""

from __future__ import annotations

import gzip
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .molpla_prep_bridge import (
    Decomposition,
    RGroupInfo,
    build_instance,
    hydrate,
    path_for,
    sample_islinked,
)

__all__ = ["MolPLAtteSample", "MolPLAtteDataset"]

#: How many neighbouring indices to try when a record is missing or torn (an
#: rsync mid-flight, a killed build).  Mirrors MolDAM's skip loop.
_MAX_SKIP = 64

#: Separate, much larger budget for indices skipped because the common-R-group
#: filter rejected them.  These are two different failures and conflating them
#: crashed a run: consecutive items in the decomposition-unit index are
#: decompositions of the SAME molecule, so a molecule that only ever cleaves off
#: common R-groups fails every one of its ~8 entries, and a few such molecules in
#: a row exhaust a 64-item budget and kill the worker.  A missing file is a
#: corpus problem worth raising on; a filtered instance is normal operation.
_MAX_FILTER_SKIP = 4096


@dataclass
class MolPLAtteSample:
    """One MolPLA training instance, ready for :func:`collate_molplatte`."""

    G: object
    P: object
    R: List[object]
    condvec: torch.Tensor
    R_hashes: List[str]
    #: ``linker_id`` per detached R-group; both ``P`` and the matching ``R``
    #: graph carry it, which is how the collate pairs their masked joints.
    joint_linker_ids: Sequence[int]
    #: Parent-molecule atom index of each shared linker atom.  ``G`` is the
    #: parent molecule, so these index straight into ``G``.
    joint_G_atoms: Sequence[int]
    #: ``linker_id -> {atom_features, cut_bond_features, incident_features}``,
    #: the pre-mask chemistry at each joint. Empty unless the assembly head is
    #: enabled; it is derived at build time, so it costs nothing on disk.
    joint_metas: Dict[int, dict]
    islinked: Sequence[bool]
    mol_id: str
    instance_id: str
    smiles: str


class MolPLAtteDataset(Dataset):
    """Per-molecule ``.pt`` corpus with on-the-fly MolPLA view construction.

    Parameters
    ----------
    dataset_path
        Corpus directory containing ``__meta__.json`` and the sharded records.
    condvec_dim
        Width of the stored condition vectors.  Checked against the corpus's
        recorded ``condvec_dim`` so a model configured for the wrong width fails
        loudly at construction rather than silently mis-slicing at step 1.
    max_rgroups
        Skip decompositions with more R-groups than this.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        condvec_dim: int = 97,
        shuffle_condvec: bool = False,
        max_rgroups: int = 8,
        seed: Optional[int] = None,
        need_assembly_targets: bool = False,
        sampling_unit: str = "decomposition",
        common_percentile: Optional[float] = None,
        max_common_fraction: float = 0.5,
    ) -> None:
        self.root = Path(dataset_path)
        meta_path = self.root / "__meta__.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"no __meta__.json at {self.root} -- is this a built corpus?"
            )
        meta = json.loads(meta_path.read_text())

        # condvec_dim = 0 means the model does not use the condition vector at
        # all, so the corpus's stored width is irrelevant and must not be
        # enforced -- the corpus still carries condvecs, they are simply ignored.
        corpus_dim = meta.get("condvec_dim")
        if condvec_dim > 0 and corpus_dim is not None and corpus_dim != condvec_dim:
            raise ValueError(
                f"condvec_dim mismatch: model expects {condvec_dim}, corpus "
                f"{self.root} was built with {corpus_dim} "
                f"(mode {meta.get('condvec_mode')!r})"
            )

        self.layout: str = meta.get("layout", "hash3")
        self.method: str = meta.get("method", "unknown")
        self.condvec_dim = condvec_dim
        self.shuffle_condvec = bool(shuffle_condvec)
        self.max_rgroups = max_rgroups
        self.need_assembly_targets = need_assembly_targets
        self.ids: List[str] = list(meta["ids"])
        if self.shuffle_condvec:
            # Built HERE, after self.ids exists. Seeded and fixed, so the wrong
            # condition is stable across epochs rather than resampled each pass.
            import numpy as _np
            _rng = _np.random.default_rng(20260831)
            self._shuffle_perm = _rng.permutation(len(self.ids))
            logging.getLogger(__name__).warning(
                "[MolPLAtteDataset] SHUFFLE TEST ACTIVE -- condition vectors are "
                "permuted across molecules. This run measures whether the condition "
                "carries signal; it is NOT a valid model."
            )
        self._rng = random.Random(seed)

        if sampling_unit not in ("decomposition", "molecule"):
            raise ValueError(
                f"sampling_unit must be 'decomposition' or 'molecule', "
                f"got {sampling_unit!r}"
            )
        self.sampling_unit = sampling_unit

        # MolPLA's common-R-group filter (paper section 2.1). R-groups above
        # `common_percentile` of the corpus occurrence distribution are
        # "common"; an instance whose DETACHED R-groups are more than
        # `max_common_fraction` common is rejected and the islinked subset is
        # redrawn. MolPLA used the 99.99th percentile and "over half", taking
        # 1,231,364 (mol, core) tuples down to 1,054,787 instances.
        #
        # This filters INSTANCES, not the library. Dropping the frequent entries
        # from the library instead removes 67.1% of all occurrences at the 99.9th
        # percentile and makes that share of queries unscoreable rather than
        # harder -- the retrieval target space must stay complete.
        #
        # Note the rule degenerates at k=1: "more than half of one R-group is
        # common" reduces to "the R-group is common", so on a naveja corpus
        # (1.11 R-groups per decomposition) it rejects far more aggressively than
        # it did for MolPLA. Measure before trusting a setting.
        self.common_percentile = common_percentile
        self.max_common_fraction = max_common_fraction
        self._common: set = set()
        self._warned_filter_fallback = False
        if common_percentile is not None:
            self._common = self._load_common(common_percentile)
        self._mol_of_item = None
        self._decomp_of_item = None
        if sampling_unit == "decomposition":
            # Index only decompositions this dataset can actually use, so an
            # item index always maps to a valid (mol, decomp) pair -- filtering
            # later would desynchronise the mapping.
            mol_idx, dec_idx = [], []
            for mi, ks in enumerate(meta.get("n_rgroups_per_decomp", [])):
                for di, k in enumerate(ks):
                    if 1 <= k <= max_rgroups:
                        mol_idx.append(mi)
                        dec_idx.append(di)
            import numpy as _np
            self._mol_of_item = _np.asarray(mol_idx, dtype=_np.int64)
            self._decomp_of_item = _np.asarray(dec_idx, dtype=_np.int64)

    def _load_common(self, percentile: float) -> set:
        """Hashes at or above *percentile* of the corpus occurrence distribution."""
        import gzip

        import numpy as np

        path = self.root / "rgroup_counts.json.gz"
        if not path.is_file():
            raise FileNotFoundError(
                f"common-R-group filtering needs {path}, written by "
                f"enumerate_rgroups.py. Rebuild the vocabulary, or leave "
                f"common_percentile unset."
            )
        with gzip.open(path, "rt") as fh:
            counts = json.load(fh)
        if not counts:
            return set()
        arr = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
        threshold = float(np.percentile(arr, percentile))
        common = {h for h, c in counts.items() if c >= threshold}
        share = sum(counts[h] for h in common) / max(arr.sum(), 1.0)
        logging.getLogger(__name__).info(
            "[MolPLAtteDataset] common-R-group filter: %s of %s hashes at "
            "p%.2f (count >= %.0f), covering %.1f%% of occurrences; instances "
            "with >%.0f%% common detached R-groups are redrawn",
            f"{len(common):,}", f"{len(counts):,}", percentile, threshold,
            100 * share, 100 * self.max_common_fraction,
        )
        return common

    def __len__(self) -> int:
        if self.sampling_unit == "decomposition":
            return int(self._mol_of_item.shape[0])
        return len(self.ids)

    def _load(self, index: int) -> Optional[dict]:
        mol_id = self.ids[index]
        try:
            raw = torch.load(
                str(path_for(self.root, mol_id, self.layout)), weights_only=False
            )
        except (FileNotFoundError, EOFError, RuntimeError, OSError):
            return None
        return hydrate(raw)

    def __getitem__(self, index: int) -> MolPLAtteSample:
        n = len(self)
        budget = _MAX_FILTER_SKIP if self._common else _MAX_SKIP
        for step in range(budget):
            item = (index + step) % n
            if self.sampling_unit == "decomposition":
                mol_i = int(self._mol_of_item[item])
                dec_i = int(self._decomp_of_item[item])
            else:
                mol_i, dec_i = item, None
            record = self._load(mol_i)
            if record is None:
                continue
            sample = self._to_sample(record, dec_i)
            if sample is not None:
                return sample
        # Exhausted. If the filter is on, it is the likely cause, and refusing
        # to return anything would abort training over a sampling preference --
        # so fall back to an unfiltered instance and say so, once.
        if self._common:
            if not self._warned_filter_fallback:
                self._warned_filter_fallback = True
                logging.getLogger(__name__).warning(
                    "[MolPLAtteDataset] common-R-group filter rejected %d "
                    "consecutive instances near index %d; falling back to "
                    "unfiltered draws. The filter is too aggressive for this "
                    "corpus -- at k=1 it reduces to 'drop if the R-group is "
                    "common', so raise common_percentile or unset it.",
                    budget, index,
                )
            for step in range(_MAX_SKIP):
                item = (index + step) % n
                if self.sampling_unit == "decomposition":
                    mol_i = int(self._mol_of_item[item])
                    dec_i = int(self._decomp_of_item[item])
                else:
                    mol_i, dec_i = item, None
                record = self._load(mol_i)
                if record is None:
                    continue
                sample = self._to_sample(record, dec_i, apply_filter=False)
                if sample is not None:
                    return sample
        raise RuntimeError(
            f"no usable record within {budget} indices of {index} in {self.root}"
        )

    def _to_sample(self, record: dict,
                   decomp_index: Optional[int] = None,
                   apply_filter: bool = True) -> Optional[MolPLAtteSample]:
        decomps = record["decompositions"]
        chosen_index = decomp_index
        if decomp_index is not None:
            # Addressed directly: the caller already knows which core it wants.
            if decomp_index >= len(decomps):
                return None
            decomp = decomps[decomp_index]
            if not (1 <= decomp["n_rgroups"] <= self.max_rgroups):
                return None
        else:
            usable = [
                (i, d) for i, d in enumerate(decomps)
                if 1 <= d["n_rgroups"] <= self.max_rgroups
            ]
            if not usable:
                return None
            chosen_index, decomp = usable[self._rng.randrange(len(usable))]

        decomposition = Decomposition(
            core_smiles=decomp.get("core_smiles", ""),
            core_atoms=tuple(int(a) for a in decomp["core_atoms"]),
            rgroups=tuple(
                RGroupInfo(
                    tuple(int(a) for a in r["rgroup_atoms"]),
                    int(r["core_linker"]),
                    int(r["rgroup_linker"]),
                )
                for r in decomp["rgroups"]
            ),
        )
        n_rg = len(decomposition.rgroups)
        hashes = decomp.get("rgroup_hashes") or []
        islinked = None
        # Redraw a few times: for k >= 2 a different subset may keep enough
        # non-common R-groups, so rejecting the instance outright would discard
        # usable variants.
        use_filter = self._common and apply_filter
        for _ in range(8 if use_filter else 1):
            cand = sample_islinked(n_rg, self._rng)
            if not use_filter:
                islinked = cand
                break
            detached = [i for i, keep in enumerate(cand) if not keep]
            if not detached:
                continue
            n_common = sum(
                1 for i in detached
                if i < len(hashes) and hashes[i] in self._common
            )
            if n_common / len(detached) <= self.max_common_fraction:
                islinked = cand
                break
        if islinked is None:
            return None

        try:
            instance = build_instance(
                record["original"],
                decomposition,
                islinked,
                mol_id=record["mol_id"],
                # Without this every instance id reads "#0" regardless of which
                # core produced it -- the id would not identify the instance,
                # and prediction tables and retrieval rows key off it.
                decomp_idx=int(chosen_index or 0),
                store_orig=self.need_assembly_targets,
                compute_hashes=False,
            )
        except Exception:
            return None

        detached = instance.detached_indices
        condvecs = torch.as_tensor(decomp["rgroup_condvecs"])[list(detached)].float()
        if self.shuffle_condvec:
            # SHUFFLE TEST. Replace this instance's condition with one drawn from
            # a different molecule, deterministically by instance index so the
            # permutation is fixed across epochs rather than resampled.
            #
            # Read against the unshuffled baseline:
            #   unchanged retrieval  -> the condition is being IGNORED; the bits
            #                           are dead weight and should be dropped.
            #   large collapse       -> the condition was carrying information the
            #                           model could not otherwise get. For a
            #                           molecule-level flavor vector that is the
            #                           intended behaviour; for anything derived
            #                           from the R-group it means LEAKAGE.
            # This single diagnostic would have caught the original 97-bit
            # fragment condvec immediately instead of after a full seed sweep.
            # Key off the molecule id, not a loop index: _to_sample has no
            # positional index in scope, and an id-derived key also keeps the
            # permutation stable regardless of sampling order.
            _k = abs(hash(record.get("mol_id", ""))) % max(len(self._shuffle_perm), 1)
            other = int(self._shuffle_perm[_k])
            try:
                alt = self._load(other)
                pool = torch.as_tensor(alt["decompositions"][0]["rgroup_condvecs"]).float()
                if pool.numel():
                    row = pool[_k % pool.shape[0]]
                    condvecs = row.unsqueeze(0).repeat(condvecs.shape[0], 1)
            except (KeyError, IndexError, FileNotFoundError):
                pass
        hashes = [decomp["rgroup_hashes"][i] for i in detached]

        return MolPLAtteSample(
            G=instance.G,
            P=instance.P,
            R=instance.R,
            condvec=condvecs,
            R_hashes=hashes,
            joint_linker_ids=instance.joint_linker_ids,
            joint_G_atoms=instance.joint_G_atoms,
            joint_metas=getattr(instance.P, "linker_metas", {}) or {},
            islinked=instance.islinked,
            mol_id=record["mol_id"],
            instance_id=instance.instance_id,
            smiles=record.get("smiles", ""),
        )
