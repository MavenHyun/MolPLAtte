"""Synt-On fragment decomposer (Yevheniia Chuiko et al.).

Synt-On applies a curated set of reaction-based SMARTS rules to
identify cleavable bonds. We collect every bond it nominates (from
2-synthon pathways) and feed them into the shared
:func:`bonds_to_partitions` converter.

Multi-cut pathways (≥3 synthons emitted by Synt-On in one pathway)
remain skipped — interpreting atom-map labels across more than two
synthons is ambiguous. Multi-cut Type 2 behaviour comes from
``bonds_to_partitions`` combining single-cut bonds combinatorially
(when ``max_cuts != None``) or all at once (when ``max_cuts=None``).
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
from typing import List, Optional, Tuple

from rdkit import Chem

from ..fragment_types import FragmentPartition
from .fragment_common import bonds_to_partitions


@contextlib.contextmanager
def _silence_io():
    devnull = io.StringIO()
    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        yield


_SYNTHON_LABEL_RE = re.compile(r":\d+")
# molpallete_prep/decomposers/synton.py -> molpallete_prep/vendor/synton
_VENDOR_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vendor", "synton",
)
_VENDOR_SRC = os.path.join(_VENDOR_ROOT, "src")
_SETUP_XML = os.path.join(_VENDOR_ROOT, "config", "Setup.xml")
_MACRO_XML = os.path.join(_VENDOR_ROOT, "config", "SetupForMacrocycles.xml")

_fragmentor = None


def _get_fragmentor():
    global _fragmentor
    if _fragmentor is not None:
        return _fragmentor
    if _VENDOR_SRC not in sys.path:
        sys.path.insert(0, _VENDOR_SRC)
    try:
        from SyntOn import fragmentation
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Synt-On adapter cannot import vendored module: {e}. "
            f"Expected source at {_VENDOR_SRC}"
        ) from e
    with _silence_io():
        _fragmentor = fragmentation(
            fragmentationMode="use_all",
            reactionsToWorkWith="R1-R13",
            maxNumberOfReactionCentersPerFragment=3,
            MaxNumberOfStages=5,
            SynthLibrary=None,
            setupFile=_SETUP_XML,
            macroCycleSetupFile=_MACRO_XML,
        )
    return _fragmentor


def _strip_labels(smiles: str) -> str:
    return _SYNTHON_LABEL_RE.sub("", smiles)


def _labeled_atom_indices(smiles: str) -> List[int]:
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        return []
    return [a.GetIdx() for a in m.GetAtoms() if a.GetAtomMapNum() > 0]


def _matches_in_mol(mol: Chem.Mol, fragment_smiles_labeled: str
                    ) -> List[Tuple[int, ...]]:
    stripped = _strip_labels(fragment_smiles_labeled)
    frag = Chem.MolFromSmiles(stripped)
    if frag is None:
        try:
            frag = Chem.MolFromSmiles(stripped, sanitize=False)
            Chem.SanitizeMol(frag)
        except Exception:
            return []
    try:
        return list(mol.GetSubstructMatches(frag, uniquify=True, useChirality=False))
    except Exception:
        return []


def _pick_partition(mol: Chem.Mol,
                     matches_a: List[Tuple[int, ...]],
                     matches_b: List[Tuple[int, ...]]
                     ) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    n = mol.GetNumAtoms()
    for ma in matches_a:
        sa = set(ma)
        if len(sa) != len(ma):
            continue
        for mb in matches_b:
            sb = set(mb)
            if sa & sb:
                continue
            if len(sa) + len(sb) != n:
                continue
            return ma, mb
    return None


def _synton_cut_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
    try:
        smi = Chem.MolToSmiles(mol)
    except Exception:
        return []
    fragmentor = _get_fragmentor()
    try:
        from SyntOn import fragmentMolecule
        with _silence_io():
            pathways, _synthons = fragmentMolecule(smi, fragmentor)
    except Exception:
        return []
    bonds: List[Tuple[int, int]] = []
    seen: set = set()
    for _, p in pathways.items():
        if getattr(p, "reagentsNumber", 0) != 2:
            continue
        synthons = list(p.participatingSynthon)
        if len(synthons) != 2:
            continue
        s_a, s_b = synthons[0].smiles, synthons[1].smiles
        la = _labeled_atom_indices(s_a); lb = _labeled_atom_indices(s_b)
        if len(la) != 1 or len(lb) != 1:
            continue
        ma = _matches_in_mol(mol, s_a); mb = _matches_in_mol(mol, s_b)
        if not ma or not mb:
            continue
        partition = _pick_partition(mol, ma, mb)
        if partition is None:
            continue
        match_a, match_b = partition
        link_a = match_a[la[0]]; link_b = match_b[lb[0]]
        bond = mol.GetBondBetweenAtoms(link_a, link_b)
        if bond is None or bond.IsInRing():
            continue
        key = (min(link_a, link_b), max(link_a, link_b))
        if key in seen:
            continue
        seen.add(key)
        bonds.append((link_a, link_b))
    return bonds


def decompose_synton(mol: Chem.Mol,
                     max_cuts: Optional[int] = None,
                     ) -> List[FragmentPartition]:
    """Synt-On cleavable bonds → fragment partitions."""
    bonds = _synton_cut_bonds(mol)
    return bonds_to_partitions(mol, bonds, max_cuts=max_cuts)
