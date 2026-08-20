"""RECAP fragment decomposer (Type 2).

Uses :class:`rdkit.Chem.Recap` to enumerate the bonds RECAP marks as
cleavable, then materialises :class:`FragmentPartition` objects.

RECAP's children-based API doesn't directly expose the bond list, so we
recurse over the cleavage tree, recording the cut bond at each split.
The classic 11 RECAP rules are encoded by RDKit itself; here we just
collect the resulting cuts.
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import Recap

from ..fragment_types import FragmentPartition
from .fragment_common import bonds_to_partitions


# Runaway guard. Distinct RECAP hierarchy nodes visited per molecule; the walk
# stops early once this many have been processed. 4096 is far above what any
# well-behaved molecule needs (the 99.9th percentile of COCONUT is under 200).
_MAX_RECAP_NODES = 4096


def _recap_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
    """Return every bond in M that RECAP nominates for cleavage.

    RECAP tags cut sites with dummy atoms. We rebuild the substructure
    of each immediate-child fragment, substruct-match it back into M
    (atom indices of the dummy atoms' neighbours = cut atoms), then
    recurse on the children for nested splits.
    """
    bonds: Set[Tuple[int, int]] = set()
    tree = Recap.RecapDecompose(mol)

    # RecapDecompose returns a DAG, not a tree: a fragment reachable by cutting
    # bonds {a, b} is the same node whether a or b was cut first, so RDKit
    # shares it between both parents. Walking it as a tree therefore re-expands
    # every node once per PATH that reaches it, which is exponential in the
    # number of cut bonds -- and each visit redid a GetSubstructMatches. On
    # peracetylated polyphenols (very common in COCONUT: many identical
    # OC(C)=O groups make the DAG maximally shared) this did not terminate.
    #
    # A fragment contributes the same cut bonds no matter which path reached it,
    # so keying on the child SMILES and visiting each distinct node once is
    # exactly equivalent and collapses the cost to the node count.
    seen: Set[str] = set()

    def walk(node):
        if len(seen) >= _MAX_RECAP_NODES:
            return
        for child_smi, child in node.children.items():
            if child_smi in seen:
                continue
            seen.add(child_smi)
            if len(seen) >= _MAX_RECAP_NODES:
                return
            try:
                child_mol = Chem.MolFromSmiles(child_smi)
            except Exception:
                continue
            if child_mol is None:
                continue
            # Get the atom-level mapping back to mol via substruct match.
            # The dummy atoms in child mark cut sites; the neighbours of
            # those dummies are the "core-side" atoms of the cuts.
            # We can't trivially recover the rgroup-side atom from a child
            # alone, so instead we walk RDKit's own atom map via
            # GetSubstructMatch on the parent.
            # Simpler approach: find dummies in child, then for each match
            # of the dummy-stripped child in M, find which M-bonds are
            # severed at the dummy positions.
            stripped = Chem.RWMol()
            old_to_new = {}
            dummy_neighbours_in_child: List[int] = []
            for atom in child_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:  # dummy
                    for nb in atom.GetNeighbors():
                        dummy_neighbours_in_child.append(nb.GetIdx())
                    continue
                old_to_new[atom.GetIdx()] = stripped.AddAtom(Chem.Atom(atom))
            for bond in child_mol.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a not in old_to_new or b not in old_to_new:
                    continue
                if stripped.GetBondBetweenAtoms(old_to_new[a], old_to_new[b]) is None:
                    stripped.AddBond(old_to_new[a], old_to_new[b], bond.GetBondType())
            try:
                Chem.SanitizeMol(stripped)
            except Exception:
                pass
            matches = mol.GetSubstructMatches(stripped, uniquify=True, useChirality=False)
            for match in matches:
                # For each dummy in child, the neighbour-in-child maps to
                # a position in match. The corresponding M-atom is the
                # core-side of the cut. The M-atom on the rgroup-side
                # is one of its neighbours in M NOT in the match's atom set.
                match_set = set(match)
                for dn_in_child in dummy_neighbours_in_child:
                    new_dn = old_to_new.get(dn_in_child)
                    if new_dn is None:
                        continue
                    if new_dn >= len(match):
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
            walk(child)

    walk(tree)
    return sorted(bonds)


def decompose_recap(mol: Chem.Mol,
                    max_cuts: Optional[int] = None,
                    ) -> List[FragmentPartition]:
    """RECAP cleavable bonds → fragment partitions."""
    bonds = _recap_bonds(mol)
    return bonds_to_partitions(mol, bonds, max_cuts=max_cuts)
