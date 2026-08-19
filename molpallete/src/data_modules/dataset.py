"""Corpus reader and ``(decomposition, islinked)`` sampler.

The corpus stores, per molecule, the intact graph plus every decomposition's
*bookkeeping* -- core atom indices and per-R-group atom indices, linkers, WL
hashes and condition vectors -- but **not** the detached graphs.  Materialising
all ``2**k - 1`` islinked subsets on disk would multiply the corpus by ~9x at the
measured 2.69 mean R-groups per decomposition, so the detach happens here, once
per ``__getitem__``.

Sampling semantics follow MolDAM: ``__len__`` is the number of *molecules*, and
each ``__getitem__`` draws one decomposition and then one islinked pattern.  Over
epochs a molecule is seen under many different (core, subset) framings.  The
per-molecule variant count is ``sum_d (2**k_d - 1)``, which at the measured
7.27 decompositions x 2.69 R-groups is on the order of 40 variants; budget
epochs accordingly (``P(seen after T epochs) ~ 1 - ((v-1)/v)**T``).
"""

from __future__ import annotations

import json
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

__all__ = ["MolPalleteSample", "MolPalleteDataset"]

#: How many neighbouring indices to try when a record is missing or torn (an
#: rsync mid-flight, a killed build).  Mirrors MolDAM's skip loop.
_MAX_SKIP = 64


@dataclass
class MolPalleteSample:
    """One MolPLA training instance, ready for :func:`collate_molpallete`."""

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


class MolPalleteDataset(Dataset):
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
        max_rgroups: int = 8,
        seed: Optional[int] = None,
        need_assembly_targets: bool = False,
    ) -> None:
        self.root = Path(dataset_path)
        meta_path = self.root / "__meta__.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"no __meta__.json at {self.root} -- is this a built corpus?"
            )
        meta = json.loads(meta_path.read_text())

        corpus_dim = meta.get("condvec_dim")
        if corpus_dim is not None and corpus_dim != condvec_dim:
            raise ValueError(
                f"condvec_dim mismatch: model expects {condvec_dim}, corpus "
                f"{self.root} was built with {corpus_dim} "
                f"(mode {meta.get('condvec_mode')!r})"
            )

        self.layout: str = meta.get("layout", "hash3")
        self.method: str = meta.get("method", "unknown")
        self.condvec_dim = condvec_dim
        self.max_rgroups = max_rgroups
        self.need_assembly_targets = need_assembly_targets
        self.ids: List[str] = list(meta["ids"])
        self._rng = random.Random(seed)

    def __len__(self) -> int:
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

    def __getitem__(self, index: int) -> MolPalleteSample:
        for step in range(_MAX_SKIP):
            record = self._load((index + step) % len(self.ids))
            if record is None:
                continue
            sample = self._to_sample(record)
            if sample is not None:
                return sample
        raise RuntimeError(
            f"no usable record within {_MAX_SKIP} indices of {index} in {self.root}"
        )

    def _to_sample(self, record: dict) -> Optional[MolPalleteSample]:
        usable = [
            d
            for d in record["decompositions"]
            if 1 <= d["n_rgroups"] <= self.max_rgroups
        ]
        if not usable:
            return None
        decomp = usable[self._rng.randrange(len(usable))]

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
        islinked = sample_islinked(len(decomposition.rgroups), self._rng)

        try:
            instance = build_instance(
                record["original"],
                decomposition,
                islinked,
                mol_id=record["mol_id"],
                store_orig=self.need_assembly_targets,
                compute_hashes=False,
            )
        except Exception:
            return None

        detached = instance.detached_indices
        condvecs = torch.as_tensor(decomp["rgroup_condvecs"])[list(detached)].float()
        hashes = [decomp["rgroup_hashes"][i] for i in detached]

        return MolPalleteSample(
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
