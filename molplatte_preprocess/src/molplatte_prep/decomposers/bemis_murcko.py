"""Bemis-Murcko scaffold decomposer (anchored paradigm).

Each molecule contributes one Decomposition: the core is the
Bemis-Murcko scaffold (rings + linkers between rings) and the R-groups
are every side chain hanging off the scaffold, partitioned by
connectivity.

Acyclic molecules (no scaffold) and molecules whose scaffold is the
whole molecule (no detachable side chains) yield no Decomposition.

Renamed entry point ``decompose_bemis_murcko`` (was ``decompose_murcko``
in the legacy ``molplatte_prep.decomposers.murcko``).
"""
from __future__ import annotations

from typing import List, Optional

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..decompose import Decomposition
from .anchored_common import decomp_from_core_atoms, filter_safe


def decompose_bemis_murcko(mol: Chem.Mol,
                            ratio: Optional[float] = None,
                            ) -> List[Decomposition]:
    """One Decomposition with core = Murcko scaffold, R-groups = side chains."""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return []
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return []

    matches = mol.GetSubstructMatches(scaffold, uniquify=True, useChirality=False)
    if not matches:
        return []

    smi = Chem.MolToSmiles(scaffold)
    out: List[Decomposition] = []
    seen = set()
    for m in matches:
        key = tuple(sorted(m))
        if key in seen:
            continue
        seen.add(key)
        d = decomp_from_core_atoms(mol, m, ratio=ratio, core_smiles=smi)
        if d is not None:
            out.append(d)
    return filter_safe(mol, out)
