"""Streaming readers for the MolPLAtte source corpora.

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
import gzip
import io
import logging
import json
import os
import pickle
import random
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

__all__ = [
    "SourceRecord",
    "SizeFilter",
    "read_flavordb",
    "read_coconut",
    "read_crossdocked",
    "read_tastepocket",
    "read_zinc",
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


#: Fields lifted from the COCONUT CSV export onto each record's ``meta``.
#: Deliberately a subset: the export has 44 columns and ~739K rows, so keeping
#: all of them would cost roughly a gigabyte of resident dict for information
#: no downstream step reads. ``organisms``/``dois``/``synonyms`` are excluded
#: for the same reason -- they are long free-text and by far the largest
#: columns. Add a field here if something needs it.
COCONUT_META_FIELDS: Tuple[str, ...] = (
    "name",
    "standard_inchi_key",
    "molecular_formula",
    "chemical_super_class",
    "chemical_class",
    "chemical_sub_class",
    "np_classifier_pathway",
    "np_classifier_superclass",
    "np_classifier_class",
    "np_likeness",
    "annotation_level",
)


def _coconut_metadata(csv_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """``base identifier -> {field: value}`` from the COCONUT CSV export.

    The 3D SDF export ships exactly one property per record (``IDENTIFIER``),
    so a corpus built from it alone carries no name, no formula and no chemical
    class. The CSV export of the same release has 44 columns; joining it back
    on the identifier recovers that without re-deriving anything.

    Accepts either the ``.zip`` as downloaded or an unpacked ``.csv``. Returns
    ``{}`` when the file is absent, so the SDF remains usable on its own.

    Keys are the BASE identifier (``CNP0252853``, not ``CNP0252853.1``) to match
    what :func:`read_coconut` yields under ``dedup_variants``. Where several
    variants of one compound appear, the first wins -- they are conformer and
    stereo variants of the same structure, so the metadata is identical.
    """
    if csv_path is None or not Path(csv_path).exists():
        return {}
    csv_path = Path(csv_path)

    # The export has fields far larger than csv's 128K default (InChI strings,
    # organism lists), which raises rather than truncating.
    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):          # 32-bit platforms
        csv.field_size_limit(2 ** 31 - 1)

    def _rows(handle):
        reader = csv.DictReader(handle)
        for row in reader:
            yield row

    out: Dict[str, Dict[str, str]] = {}
    try:
        if csv_path.suffix == ".zip":
            with zipfile.ZipFile(csv_path) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not names:
                    return {}
                with zf.open(names[0]) as fh:
                    stream = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
                    for row in _rows(stream):
                        _absorb(out, row)
        else:
            with csv_path.open(newline="", encoding="utf-8", errors="replace") as fh:
                for row in _rows(fh):
                    _absorb(out, row)
    except (OSError, csv.Error, zipfile.BadZipFile):
        # Metadata is an enrichment, never a prerequisite: a truncated or
        # corrupt export must not stop a corpus build.
        return out
    return out


def _absorb(out: Dict[str, Dict[str, str]], row: Dict[str, str]) -> None:
    ident = (row.get("identifier") or "").strip()
    if not ident:
        return
    base = ident.split(".")[0]
    if base in out:
        return
    out[base] = {
        f: v.strip()
        for f in COCONUT_META_FIELDS
        if (v := (row.get(f) or "")).strip()
    }


def read_coconut(
    path: str | Path,
    size_filter: Optional[SizeFilter] = None,
    limit: Optional[int] = None,
    dedup_variants: bool = True,
    metadata_csv: Optional[str | Path] = None,
) -> Iterator[SourceRecord]:
    """Stream COCONUT, recomputing SMILES from each 3D molblock.

    Parameters
    ----------
    dedup_variants
        Collapse ``CNP0252853.1`` / ``.2`` / ... to one record per base ID.  The
        suffix indexes conformer and stereo variants of the same compound, and
        keeping them all would triple-count 250K compounds in the R-group
        statistics.  Kept as a flag because a stereo-aware study might want them.

    metadata_csv
        COCONUT CSV export (``.zip`` or ``.csv``) of the SAME release, joined on
        identifier to attach name, formula and chemical class. Defaults to
        ``coconut_csv-<release>.zip`` beside the SDF; pass ``False``-y to skip.
        Absent file is not an error -- the SDF alone still builds.

    Yields ``mol_id = "CNP..."`` (the base identifier).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"COCONUT SDF not found at {path}")

    sf = size_filter or SizeFilter()
    seen: Set[str] = set()
    emitted = 0

    if metadata_csv is None:
        guess = path.parent / f"coconut_csv-{path.stem.split('-', 1)[-1]}.zip"
        metadata_csv = guess if guess.exists() else None
    meta_by_id = _coconut_metadata(metadata_csv) if metadata_csv else {}
    if meta_by_id:
        logging.getLogger(__name__).info(
            "[read_coconut] joined metadata for %s compounds from %s",
            f"{len(meta_by_id):,}", Path(metadata_csv).name,
        )

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

        meta = {"identifier": identifier}
        meta.update(meta_by_id.get(base_id, {}))
        yield SourceRecord(base_id, smiles, "coconut", meta)
        emitted += 1
        if limit is not None and emitted >= limit:
            break


#: ``name -> (reader, default path)``.  Paths are overridable on the CLI.
def _xd_split(keys: Sequence[str], split_of: Dict[int, str]) -> str:
    """Split label for a ligand spanning several pocket records.

    ``test`` wins over ``train`` wins over ``unassigned``: a ligand that appears
    against any held-out pocket must not be trainable, or the CrossDocked test
    set leaks through ligand-level deduplication.
    """
    seen = {split_of.get(int(k), "unassigned") for k in keys if k.isdigit()}
    for level in ("test", "val", "train"):
        if level in seen:
            return level
    return "unassigned"


def read_crossdocked(
    path: str | Path,
    size_filter: Optional[SizeFilter] = None,
    limit: Optional[int] = None,
    split_path: Optional[str | Path] = None,
    drug_like: bool = True,
    keep_cofactors: bool = False,
) -> Iterator[SourceRecord]:
    """Stream CrossDocked2020 ligands from the processed ``pocket10`` LMDB.

    ONE RECORD PER DISTINCT LIGAND, not per pocket-ligand pair. The LMDB holds
    166,500 records but only 11,735 distinct ligand SMILES -- CrossDocked is a
    CROSS-docking set, so the mean ligand appears against 14.2 different pockets
    and one appears against 1,100. Emitting pairs would multiply every R-group's
    corpus count by that factor, unevenly, which corrupts two things that read
    those counts: the frequency prior that Hit@K is judged against, and the
    logQ correction, whose whole job is to subtract log p(k).

    The pocket side is not discarded -- each record carries the full list of
    LMDB keys whose pocket binds it, so the pocket-conditioning stage can expand
    one ligand back into its pairs without re-reading the LMDB.

    meta
        ``n_pockets``      how many pockets bind this ligand
        ``pocket_keys``    comma-joined LMDB keys for those pairs
        ``protein_file``   representative pocket (first key's protein_filename)
        ``ligand_file``    representative ligand path inside CrossDocked
        ``split``          train/test/unassigned, from the pose split file

    drug_like
        Drop crystallographic artifacts -- cryoprotectants, buffers, ions,
        detergents and nucleotide cofactors -- identified by the PDB chemical
        component code embedded in ``ligand_filename``. Removes 4.0% of distinct
        ligands but 11.3% of pocket pairs, because artifacts are exactly the
        high-reuse head (ADP appears against 1,903 pockets, SAH 1,121, the MRD
        cryoprotectant 640). See :mod:`molplatte_prep.pocket_ligands`.

    keep_cofactors
        Re-admit ATP/NAD/SAH and friends. They are genuine cognate ligands for
        the enzymes that use them, so this is a real choice rather than a
        loosening -- but they bind almost everything, so they are excluded by
        default.

    Yields ``mol_id = "XD<first-lmdb-key>"``.
    """
    from .pocket_ligands import assess_ligand, ccd_code_from_path
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CrossDocked LMDB not found at {path}")
    try:
        import lmdb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("read_crossdocked needs `lmdb` (pip install lmdb)") from exc

    split_of: Dict[int, str] = {}
    sp = Path(split_path) if split_path else path.parent / "crossdocked_pocket10_pose_split.pt"
    if sp.is_file():
        try:
            import torch

            for name, idxs in torch.load(sp, weights_only=False).items():
                for i in idxs:
                    split_of[int(i)] = name
        except Exception as exc:  # noqa: BLE001
            logging.warning("[crossdocked] could not read split %s: %s", sp, exc)

    sf = size_filter or SizeFilter()
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=False, max_readers=64)

    # pass 1: group pocket keys by ligand SMILES, holding only strings
    order: List[str] = []
    keys_of: Dict[str, List[str]] = {}
    files_of: Dict[str, Tuple[str, str]] = {}
    with env.begin() as txn:
        for raw_key, raw_val in txn.cursor():
            try:
                rec = pickle.loads(raw_val)
            except Exception:
                continue
            smi = (rec.get("ligand_smiles") or "").strip()
            if not smi:
                continue
            k = raw_key.decode()
            if smi not in keys_of:
                keys_of[smi] = []
                order.append(smi)
                files_of[smi] = (str(rec.get("protein_filename") or ""),
                                 str(rec.get("ligand_filename") or ""))
            keys_of[smi].append(k)
    env.close()
    logging.info("[crossdocked] %s pockets over %s distinct ligands (%.1fx reuse)",
                 f"{sum(len(v) for v in keys_of.values()):,}", f"{len(order):,}",
                 sum(len(v) for v in keys_of.values()) / max(len(order), 1))

    emitted = 0
    dropped: Dict[str, int] = {}
    for smi in order:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not sf.accepts(mol):
            continue
        ks = keys_of[smi]
        prot, lig = files_of[smi]
        code = ccd_code_from_path(lig)
        if drug_like:
            verdict = assess_ligand(
                mol, code, keep_cofactors=keep_cofactors,
                min_heavy_atoms=sf.min_heavy_atoms,
                max_heavy_atoms=sf.max_heavy_atoms,
            )
            if not verdict.ok:
                dropped[verdict.reason] = dropped.get(verdict.reason, 0) + 1
                continue
        first = ks[0]
        yield SourceRecord(
            mol_id=f"XD{first}",
            smiles=smi,
            source="crossdocked",
            meta={
                "n_pockets": str(len(ks)),
                "pocket_keys": ",".join(ks),
                "protein_file": prot,
                "ligand_file": lig,
                # Split over ALL of this ligand's pockets, with test taking
                # precedence. 86 ligands bind both train and test pockets, and
                # deduplicating to one record per ligand would otherwise place
                # them in train and leak the official test set.
                "split": _xd_split(ks, split_of),
                "ccd": code or "",
            },
        )
        emitted += 1
        if limit is not None and emitted >= limit:
            return

    if dropped:
        logging.info("[crossdocked] dropped %s non-drug-like ligands: %s",
                     f"{sum(dropped.values()):,}",
                     ", ".join(f"{k} {v}" for k, v in
                               sorted(dropped.items(), key=lambda x: -x[1])))


def read_tastepocket(
    path: str,
    *,
    limit: Optional[int] = None,
    size_filter: Optional[SizeFilter] = None,
    **_: object,
) -> Iterator[SourceRecord]:
    """Stream the tastepocket fine-tuning set built by build_tastepocket_dataset.py.

    ONE RECORD PER (LIGAND, RECEPTOR), already aggregated upstream. Do not
    re-expand to pocket instances: 1,255 sites collapse to 269 records because a
    homotetramer deposits the same ligand four times, and counting those
    separately inflates the R-group frequency prior that Hit@K is judged against
    and that the logQ correction subtracts.

    Each record carries its ESM-2 pocket embedding RAW, at the language model's
    own width. ``StoredPocketCondVec`` hands it through unreduced;
    ``PocketConditioning`` learns the reduction inside the model.

    meta
        ``ccd``                PDB chemical component id
        ``uniprot``            receptor identity the fold was cut on
        ``fold``               CV fold, 0-4; the final checkpoint ignores it
        ``component``          bipartite component the fold was cut from
        ``families``           receptor families, semicolon-joined
        ``flavor_labels``      resolved sensory labels, semicolon-joined
        ``flavor_source``      measured / mined / llm / llm_abstain
        ``pdb_ids``            entries this (ligand, receptor) pair came from
        ``pocket_embedding``   the raw pocket vector
        ``InChIKey``           joins the flavor tables
    """
    n = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            smiles = rec.get("smiles") or ""
            if not smiles:
                continue
            if size_filter is not None:
                mol = Chem.MolFromSmiles(smiles)
                if not size_filter.accepts(mol):
                    continue
            yield SourceRecord(
                mol_id=rec["id"],
                smiles=smiles,
                source="tastepocket",
                meta={
                    "ccd": rec.get("ccd", ""),
                    "uniprot": rec.get("uniprot", ""),
                    "fold": str(rec.get("fold", "")),
                    "component": str(rec.get("component", "")),
                    "families": ";".join(rec.get("families") or []),
                    "cats": ";".join(rec.get("cats") or []),
                    "organisms": ";".join(rec.get("organisms") or []),
                    "pdb_ids": ";".join(rec.get("pdb_ids") or []),
                    "flavor_labels": ";".join(rec.get("flavor_labels") or []),
                    "flavor_source": rec.get("flavor_source", ""),
                    "n_instances": str(rec.get("n_instances", "")),
                    "InChIKey": rec.get("inchikey", ""),
                    "name": rec.get("name") or "",
                    "pocket_embedding": rec.get("pocket_embedding") or [],
                },
            )
            n += 1
            if limit and n >= limit:
                return


#: ZINC encodes chemistry in its directory names: the first letter of a tranche
#: is a molecular-weight bin, the second a logP bin. Measured on this download:
#: B*~233 Da, C*~283, D*~315, E*~339, F*~363, G*~388, H*~414, I*~437, J*~475;
#: logP runs *A~-2.5 to *J~+3.4.
ZINC_TRANCHE_MW = {"B": 233, "C": 283, "D": 315, "E": 339, "F": 363,
                   "G": 388, "H": 414, "I": 437, "J": 475}

#: Measured over a 470k-molecule sample of this archive. Used only to size the
#: per-tranche quota before reading; the actual draw counts real molecules.
_ZINC_BYTES_PER_MOL = 693.0


def _zinc_quotas(tranches: Dict[str, int], total: int) -> Dict[str, int]:
    """Split *total* equally across tranches, capping those that cannot fill it.

    An equal split is the point: ZINC's natural distribution puts 52% of its
    mass in the D and E rows around 315-339 Da, and only 2.8% in the B row at
    233 Da -- the row closest to flavour chemistry. Sampling proportionally
    would reproduce that skew and pretrain on a narrow drug-like band.

    12 of the 90 tranches hold less than an equal share (``FD`` is 492 KB in
    total), so their shortfall is redistributed over the tranches that still
    have room, repeatedly, until either the target is met or every tranche is
    at capacity. Without that, an equal split silently returns fewer molecules
    than asked for.
    """
    quota = {t: 0 for t in tranches}
    remaining = dict(tranches)
    want = total
    while want > 0:
        open_t = [t for t in remaining if remaining[t] > 0]
        if not open_t:
            break
        share = max(want // len(open_t), 1)
        moved = 0
        for t in open_t:
            take = min(share, remaining[t], want - moved)
            if take <= 0:
                continue
            quota[t] += take
            remaining[t] -= take
            moved += take
            if moved >= want:
                break
        if moved == 0:
            break
        want -= moved
    return quota


def read_zinc(
    path: str | Path,
    *,
    size_filter: Optional[SizeFilter] = None,
    limit: Optional[int] = None,
    total: int = 10_000_000,
    seed: int = 20260908,
    **_: object,
) -> Iterator[SourceRecord]:
    """Stream a tranche-balanced sample of ZINC2020 3D SDF shards.

    ``total`` molecules are drawn EQUALLY across the 90 tranches, with the
    shortfall from tranches too small to fill their share redistributed to
    those that can (see :func:`_zinc_quotas`). Within a tranche, shards are
    shuffled and read until the quota is met, so the draw is not biased toward
    whichever shards happen to sort first.

    No condition vector is available or implied: ZINC has neither flavour
    annotations nor pockets. Every record therefore carries an empty ``meta``
    beyond provenance, and the corpus should be built with
    ``--condvec-mode neutral`` or a zeroed flavour vector.

    ``limit`` is an absolute cap applied after quotas, for smoke tests.
    """
    root = Path(path)
    shards_by_tranche: Dict[str, List[Path]] = {}
    for shard in root.rglob("*.sdf.gz"):
        if shard.stat().st_size <= 0:
            continue
        tranche = shard.relative_to(root).parts[0]
        shards_by_tranche.setdefault(tranche, []).append(shard)
    if not shards_by_tranche:
        raise FileNotFoundError(f"no .sdf.gz shards under {root}")

    capacity = {
        t: int(sum(s.stat().st_size for s in sh) / _ZINC_BYTES_PER_MOL)
        for t, sh in shards_by_tranche.items()
    }
    quotas = _zinc_quotas(capacity, total)
    logging.info(
        "[zinc] %d tranches, ~%.0fM available, drawing %s equally "
        "(%d tranches capped below their share)",
        len(quotas), sum(capacity.values()) / 1e6, f"{total:,}",
        sum(1 for t in quotas if quotas[t] >= capacity[t] and capacity[t] > 0),
    )

    rng = random.Random(seed)
    sf = size_filter or SizeFilter()
    emitted = 0
    for tranche in sorted(shards_by_tranche):
        want = quotas.get(tranche, 0)
        if want <= 0:
            continue
        shards = list(shards_by_tranche[tranche])
        rng.shuffle(shards)
        got = 0
        for shard in shards:
            if got >= want:
                break
            try:
                with gzip.open(shard, "rb") as fh:
                    supplier = Chem.ForwardSDMolSupplier(
                        fh, removeHs=True, sanitize=True)
                    for mol in supplier:
                        if got >= want:
                            break
                        if mol is None or not sf.accepts(mol):
                            continue
                        try:
                            smiles = Chem.MolToSmiles(mol)
                        except Exception:  # noqa: BLE001
                            continue
                        if not smiles:
                            continue
                        zid = (mol.GetProp("_Name").strip()
                               if mol.HasProp("_Name") else f"ZINC{emitted}")
                        yield SourceRecord(zid, smiles, "zinc",
                                           {"tranche": tranche,
                                            "tranche_mw_bin": tranche[0],
                                            "tranche_logp_bin": tranche[1:],
                                            "shard": shard.name})
                        got += 1
                        emitted += 1
                        if limit is not None and emitted >= limit:
                            return
            except Exception as exc:  # noqa: BLE001
                # One corrupt shard must not end a 10M-molecule draw.
                logging.warning("[zinc] skipping %s: %s", shard.name, exc)
                continue
        if got < want:
            logging.warning("[zinc] tranche %s yielded %s of %s requested",
                            tranche, f"{got:,}", f"{want:,}")


SOURCES: Dict[str, Tuple[object, str]] = {
    "flavordb": (read_flavordb, os.path.expanduser("~/datasets/flavordb")),
    "coconut": (
        read_coconut,
        os.path.expanduser("~/datasets/coconut/coconut_sdf_3d-08-2026.sdf"),
    ),
    "zinc": (
        read_zinc,
        os.path.expanduser("~/datasets/zinc2020/raw"),
    ),
    "tastepocket": (
        read_tastepocket,
        os.path.expanduser("~/preprocessed/molplatte/tastepocket/dataset.jsonl"),
    ),
    "crossdocked": (
        read_crossdocked,
        os.path.expanduser(
            "~/datasets/crossdocked2020/"
            "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
        ),
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
