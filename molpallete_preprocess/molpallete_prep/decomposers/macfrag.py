"""MacFrag fragment decomposer (Diao et al. 2023).

MacFrag emits fragment SMILES with ``[n*]`` dummy markers at cut points.
We collect every dummy → real-atom edge across all emitted fragments,
re-identify the corresponding bonds in M, and use them as the
cleavable-bond set for a Type 2 partition.

Lazy import of :mod:`igraph`; raises a helpful ImportError if missing.
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from rdkit import Chem

from ..fragment_types import FragmentPartition
from .fragment_common import bonds_to_partitions


def _macfrag_smis(mol: Chem.Mol,
                  max_blocks: int,
                  max_sr: int,
                  min_frag_atoms: int) -> List[str]:
    try:
        from ..vendor.macfrag import MacFrag
    except ModuleNotFoundError as e:
        if "igraph" in str(e):
            raise ImportError(
                "decompose_macfrag requires python-igraph; install with "
                "`pip install python-igraph`."
            ) from e
        raise
    # MacFrag's vendored impl mutates the input mol (it calls
    # ``atom.SetAtomMapNum(idx)`` on every atom). Pass a copy so callers
    # don't see their mol mutated.
    return MacFrag(Chem.Mol(mol), maxBlocks=max_blocks, maxSR=max_sr,
                   asMols=False, minFragAtoms=min_frag_atoms)


def _macfrag_bonds(mol: Chem.Mol, frag_smis: List[str]) -> List[Tuple[int, int]]:
    """Recover the cleavable bonds in M from MacFrag's labelled-fragment
    SMILES list.

    For each emitted fragment, strip the dummy atoms and substruct-match
    the result into M. Every dummy in the fragment maps to an M-atom
    via the matched substructure's neighbour at that position; the bond
    between that M-atom and one of its non-matched M-neighbours is a
    cut bond.
    """
    bonds: Set[Tuple[int, int]] = set()
    for smi in frag_smis:
        frag = Chem.MolFromSmiles(smi)
        if frag is None:
            continue
        # Strip dummies, track which fragment atom each dummy was bonded to.
        keep_atoms: List[int] = []
        old_to_new = {}
        dummy_neighbours: List[int] = []
        rw = Chem.RWMol()
        for atom in frag.GetAtoms():
            if atom.GetAtomicNum() == 0:
                for nb in atom.GetNeighbors():
                    dummy_neighbours.append(nb.GetIdx())
                continue
            keep_atoms.append(atom.GetIdx())
            old_to_new[atom.GetIdx()] = rw.AddAtom(Chem.Atom(atom))
        for bond in frag.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a not in old_to_new or b not in old_to_new:
                continue
            if rw.GetBondBetweenAtoms(old_to_new[a], old_to_new[b]) is None:
                rw.AddBond(old_to_new[a], old_to_new[b], bond.GetBondType())
        try:
            Chem.SanitizeMol(rw)
        except Exception:
            continue
        if rw.GetNumAtoms() == 0:
            continue
        if rw.GetNumHeavyAtoms() >= mol.GetNumHeavyAtoms():
            continue
        try:
            matches = mol.GetSubstructMatches(rw, uniquify=True, useChirality=False)
        except Exception:
            continue
        for match in matches:
            match_set = set(match)
            for dn in dummy_neighbours:
                new_dn = old_to_new.get(dn)
                if new_dn is None or new_dn >= len(match):
                    continue
                m_core = match[new_dn]
                for nb in mol.GetAtomWithIdx(m_core).GetNeighbors():
                    if nb.GetIdx() in match_set:
                        continue
                    bond_obj = mol.GetBondBetweenAtoms(m_core, nb.GetIdx())
                    if bond_obj is None or bond_obj.IsInRing():
                        continue
                    bonds.add((min(m_core, nb.GetIdx()),
                               max(m_core, nb.GetIdx())))
    return sorted(bonds)


def decompose_macfrag(mol: Chem.Mol,
                      max_cuts: Optional[int] = None,
                      max_blocks: int = 4,
                      max_sr: int = 8,
                      min_frag_atoms: int = 2,
                      ) -> List[FragmentPartition]:
    """MacFrag cleavable bonds → fragment partitions."""
    try:
        smis = _macfrag_smis(mol, max_blocks, max_sr, min_frag_atoms)
    except Exception:
        return []
    bonds = _macfrag_bonds(mol, smis)
    return bonds_to_partitions(mol, bonds, max_cuts=max_cuts)
