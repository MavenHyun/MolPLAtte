"""Streaming readers for the MolPallete source corpora.

Two sources, with quite different shapes:

FlavorDB (``~/datasets/flavordb``)
    25,595 PubChem compounds scraped from FlavorDB2, keyed by PubChem CID.
    ``properties.csv`` carries isomeric SMILES, InChIKey and descriptors for all
    of them; ``flavordb_molecules.csv`` carries the semicolon-delimited
    ``flavor_profile`` labels (98.1% populated, 715 distinct tokens).  The CSV
    route is preferred over the SDFs because ``flavordb_3d.sdf`` has no SMILES at
    all and covers only 70.3% of the set.

COCONUT (``~/datasets/coconut/coconut_sdf_3d-08-2026.sdf``)
    737,343 natural-product 3D records, 2.3 GB.  **The only SDF tag is
    ``IDENTIFIER``** -- no SMILES, no InChIKey, no taxonomy -- so chemistry is
    recomputed from the molblock.  Identifiers are ``CNP0252853.1``: the suffix
    indexes conformer/stereo variants, and the 737,343 records collapse to
    489,395 distinct base IDs, so deduplication is mandatory rather than
    optional.

Both readers stream (``ForwardSDMolSupplier`` / chunked CSV) and never hold a
whole file in memory.

Flavor-relevance filtering
--------------------------
COCONUT skews large: median 31 heavy atoms, p90 63, max 475, against FlavorDB's
median 23 / p90 45.  Natural products at the top of that range (macrolides,
polysaccharides, large glycosides) are not flavor-relevant and would dominate the
R-group vocabulary.  :class:`SizeFilter` defaults to ``[5, 50]`` heavy atoms,
which is roughly FlavorDB's 93rd percentile and drops the largest 4.4% (~32K) of
COCONUT.  ``--max-heavy-atoms 40`` gives a more volatiles-focused corpus (87% of
COCONUT); the flag exists precisely because that trade-off is a judgement call.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Set, Tuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

__all__ = [
    "SourceRecord",
    "SizeFilter",
    "read_flavordb",
    "read_coconut",
    "read_source",
    "SOURCES",
]

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


@dataclass
class SourceRecord:
    """One molecule as it comes off a source, before washing or decomposition."""

    mol_id: str
    smiles: str
    source: str
    #: Free-form provenance kept on the record (flavor labels, CID, ...).
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class SizeFilter:
    """Heavy-atom bounds, applied before the expensive decomposition step.

    ``min_heavy_atoms`` also screens out the degenerate records observed in
    COCONUT (heavy-atom count 0) and single-atom entries that cannot be cut.
    """

    min_heavy_atoms: int = 5
    max_heavy_atoms: int = 50

    def accepts(self, mol: Chem.Mol) -> bool:
        if mol is None:
            return False
        n = mol.GetNumHeavyAtoms()
        return self.min_heavy_atoms <= n <= self.max_heavy_atoms

    def as_dict(self) -> Dict[str, int]:
        return {
            "min_heavy_atoms": self.min_heavy_atoms,
            "max_heavy_atoms": self.max_heavy_atoms,
        }


def _flavor_labels(flavordb_root: Path) -> Dict[str, str]:
    """``cid -> flavor_profile`` from ``flavordb_molecules.csv``, or ``{}``."""
    path = flavordb_root / "flavordb_molecules.csv"
    if not path.is_file():
        return {}
    out: Dict[str, str] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            cid = (row.get("cid") or "").strip()
            profile = (row.get("flavor_profile") or "").strip()
            if cid and profile:
                out[cid] = profile
    return out


def read_flavordb(
    root: str | Path,
    size_filter: Optional[SizeFilter] = None,
    limit: Optional[int] = None,
) -> Iterator[SourceRecord]:
    """Stream FlavorDB from ``properties.csv``, joined to flavor labels on CID.

    Yields ``mol_id = "FDB{cid}"``.  Size filtering uses the ``HeavyAtomCount``
    column when present, which avoids parsing SMILES for records that will be
    rejected anyway.
    """
    root = Path(root)
    props = root / "properties.csv"
    if not props.is_file():
        raise FileNotFoundError(f"FlavorDB properties.csv not found at {props}")

    labels = _flavor_labels(root)
    sf = size_filter or SizeFilter()
    emitted = 0

    with props.open(newline="") as fh:
        for row in csv.DictReader(fh):
            cid = (row.get("CID") or "").strip()
            smiles = (row.get("SMILES") or "").strip()
            if not cid or not smiles:
                continue
            # Cheap pre-filter on the tabulated heavy-atom count.
            raw_hac = (row.get("HeavyAtomCount") or "").strip()
            if raw_hac:
                try:
                    hac = int(float(raw_hac))
                except ValueError:
                    hac = None
                if hac is not None and not (
                    sf.min_heavy_atoms <= hac <= sf.max_heavy_atoms
                ):
                    continue
            meta = {"cid": cid}
            if cid in labels:
                meta["flavor_profile"] = labels[cid]
            for key in ("InChIKey", "MolecularFormula", "MolecularWeight"):
                value = (row.get(key) or "").strip()
                if value:
                    meta[key] = value
            yield SourceRecord(f"FDB{cid}", smiles, "flavordb", meta)
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def read_coconut(
    path: str | Path,
    size_filter: Optional[SizeFilter] = None,
    limit: Optional[int] = None,
    dedup_variants: bool = True,
) -> Iterator[SourceRecord]:
    """Stream COCONUT, recomputing SMILES from each 3D molblock.

    Parameters
    ----------
    dedup_variants
        Collapse ``CNP0252853.1`` / ``.2`` / ... to one record per base ID.  The
        suffix indexes conformer and stereo variants of the same compound, and
        keeping them all would triple-count 250K compounds in the R-group
        statistics.  Kept as a flag because a stereo-aware study might want them.

    Yields ``mol_id = "CNP..."`` (the base identifier).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"COCONUT SDF not found at {path}")

    sf = size_filter or SizeFilter()
    seen: Set[str] = set()
    emitted = 0

    supplier = Chem.ForwardSDMolSupplier(str(path), removeHs=True, sanitize=True)
    for index, mol in enumerate(supplier):
        if mol is None:
            continue
        if not sf.accepts(mol):
            continue

        identifier = (
            mol.GetProp("IDENTIFIER") if mol.HasProp("IDENTIFIER") else f"COCONUT{index}"
        )
        base_id = identifier.split(".")[0]
        if dedup_variants:
            if base_id in seen:
                continue
            seen.add(base_id)

        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            continue
        if not smiles:
            continue

        yield SourceRecord(
            base_id, smiles, "coconut", {"identifier": identifier}
        )
        emitted += 1
        if limit is not None and emitted >= limit:
            return


#: ``name -> (reader, default path)``.  Paths are overridable on the CLI.
SOURCES: Dict[str, Tuple[object, str]] = {
    "flavordb": (read_flavordb, os.path.expanduser("~/datasets/flavordb")),
    "coconut": (
        read_coconut,
        os.path.expanduser("~/datasets/coconut/coconut_sdf_3d-08-2026.sdf"),
    ),
}


def read_source(
    name: str,
    path: Optional[str | Path] = None,
    size_filter: Optional[SizeFilter] = None,
    limit: Optional[int] = None,
    **kwargs,
) -> Iterator[SourceRecord]:
    """Dispatch to the reader named *name*."""
    if name not in SOURCES:
        raise ValueError(f"unknown source {name!r}; available: {sorted(SOURCES)}")
    reader, default_path = SOURCES[name]
    return reader(path or default_path, size_filter=size_filter, limit=limit, **kwargs)
