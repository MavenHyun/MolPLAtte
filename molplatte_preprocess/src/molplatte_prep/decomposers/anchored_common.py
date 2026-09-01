"""Shared helpers for decomposer adapters.

Every decomposer in this package ultimately yields ``Decomposition`` /
``RGroupInfo`` objects with the same semantics as the original
``molplatte_prep.decompose`` (one core, k R-groups, per-R-group linker atoms).

To keep individual adapters thin we share two converters here:

* :func:`decomp_from_bond_cut` — given one cleavable bond ``(a, b)`` in M,
  return the Decomposition placing the larger component as the core. Used
  by RECAP / BRICS / r-BRICS / ring-aware.

* :func:`decomp_from_core_atoms` — given an atom-index set in M to use as
  the core, partition the remainder into connected components and identify
  the linker bond of each. Used by Murcko / MacFrag (which directly nominate
  a sub-structure as the scaffold/core).

Both honour a Naveja-style ``ratio`` (heavy-atom share of M required of the
core). ``ratio=None`` disables the filter; ``ratio=2/3`` reproduces
MolPLA-paper behaviour.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set, Tuple

from rdkit import Chem

from ..decompose import (
    Decomposition,
    RGroupInfo,
    _connected_components,
)


def _bfs_skip_bond(adj: List[List[Tuple[int, int]]],
                   n_atoms: int,
                   skip_bond_idx: int,
                   start: int) -> List[int]:
    """BFS from ``start`` over ``adj`` skipping ``skip_bond_idx``.

    ``adj[a]`` is a list of ``(neighbour, bond_idx)`` tuples.
    """
    visited = [False] * n_atoms
    visited[start] = True
    stack = [start]
    comp = []
    while stack:
        a = stack.pop()
        comp.append(a)
        for nbr, bidx in adj[a]:
            if bidx == skip_bond_idx or visited[nbr]:
                continue
            visited[nbr] = True
            stack.append(nbr)
    return comp


def _build_adj(mol: Chem.Mol) -> List[List[Tuple[int, int]]]:
    n = mol.GetNumAtoms()
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bidx = bond.GetIdx()
        adj[i].append((j, bidx))
        adj[j].append((i, bidx))
    return adj


def decomp_from_bond_cut(mol: Chem.Mol,
                          atom_a: int,
                          atom_b: int,
                          ratio: Optional[float] = None,
                          ) -> Optional[Decomposition]:
    """Cut one bond ``a-b`` and assemble a Decomposition.

    The larger of the two resulting components is the core; if its heavy-atom
    count is at least ``ratio * nHA(M)`` (or ``ratio is None``) the
    decomposition is returned, otherwise ``None``.
    """
    bond = mol.GetBondBetweenAtoms(atom_a, atom_b)
    if bond is None:
        return None
    if bond.IsInRing():                           # cutting a ring bond is wrong
        return None
    bidx = bond.GetIdx()

    adj = _build_adj(mol)
    n = mol.GetNumAtoms()
    comp_a = _bfs_skip_bond(adj, n, bidx, atom_a)
    if atom_b in comp_a:                          # ring (shouldn't reach here)
        return None
    comp_b = _bfs_skip_bond(adj, n, bidx, atom_b)

    if len(comp_a) >= len(comp_b):
        core_atoms, rg_atoms = comp_a, comp_b
        core_linker, rgroup_linker = atom_a, atom_b
    else:
        core_atoms, rg_atoms = comp_b, comp_a
        core_linker, rgroup_linker = atom_b, atom_a

    if ratio is not None and len(core_atoms) < ratio * mol.GetNumHeavyAtoms():
        return None

    return Decomposition(
        core_smiles="",
        core_atoms=tuple(sorted(core_atoms)),
        rgroups=(RGroupInfo(
            rgroup_atoms=tuple(sorted(rg_atoms)),
            core_linker=core_linker,
            rgroup_linker=rgroup_linker,
        ),),
    )


def decomp_from_bond_cuts(mol: Chem.Mol,
                            cut_bonds: Sequence[Tuple[int, int]],
                            ratio: Optional[float] = None,
                            ) -> Optional[Decomposition]:
    """Cut **k bonds simultaneously** and build one multi-joint Decomposition.

    Generalises :func:`decomp_from_bond_cut` to k ≥ 1. Returns ``None`` if:

    - any cut bond is missing or in a ring,
    - two cuts share an atom (geminal joint — incompatible with the
      one-joint-per-template-atom model in ``detach_rgroups_multi``),
    - the largest component (core) fails the Naveja ratio threshold.

    The largest component after removing all k bonds is the core; every
    other component becomes one R-group with its linker atom identified
    via the cut bond that re-attaches it to the core.
    """
    if not cut_bonds:
        return None

    cut_bonds = tuple(cut_bonds)
    cut_atoms: Set[int] = set()
    cut_bond_idxs: Set[int] = set()
    for a, b in cut_bonds:
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None or bond.IsInRing():
            return None
        if a in cut_atoms or b in cut_atoms:
            return None                           # geminal
        cut_atoms.add(a); cut_atoms.add(b)
        cut_bond_idxs.add(bond.GetIdx())

    # Adjacency excluding the cut bonds.
    n = mol.GetNumAtoms()
    adj: List[List[int]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        if bond.GetIdx() in cut_bond_idxs:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i].append(j); adj[j].append(i)

    visited = [False] * n
    components: List[List[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        comp: List[int] = []
        stack = [start]
        while stack:
            a = stack.pop()
            if visited[a]:
                continue
            visited[a] = True
            comp.append(a)
            for nb in adj[a]:
                if not visited[nb]:
                    stack.append(nb)
        components.append(comp)

    if len(components) < 2:
        return None                               # cuts didn't disconnect anything

    components.sort(key=len, reverse=True)
    core_atoms = components[0]
    core_set: Set[int] = set(core_atoms)

    if ratio is not None and len(core_atoms) < ratio * mol.GetNumHeavyAtoms():
        return None

    rgroup_infos: List[RGroupInfo] = []
    used_core_linkers: Set[int] = set()
    for comp in components[1:]:
        comp_set = set(comp)
        linker_core = linker_rg = None
        for a, b in cut_bonds:
            if a in core_set and b in comp_set:
                linker_core, linker_rg = a, b
                break
            if b in core_set and a in comp_set:
                linker_core, linker_rg = b, a
                break
        if linker_rg is None:
            return None
        if linker_core in used_core_linkers:
            return None                           # second R-group on same core atom
        used_core_linkers.add(linker_core)
        rgroup_infos.append(RGroupInfo(
            rgroup_atoms=tuple(sorted(comp)),
            core_linker=linker_core,
            rgroup_linker=linker_rg,
        ))

    return Decomposition(
        core_smiles="",
        core_atoms=tuple(sorted(core_atoms)),
        rgroups=tuple(rgroup_infos),
    )


def decomp_from_core_atoms(mol: Chem.Mol,
                            core_atoms: Iterable[int],
                            ratio: Optional[float] = None,
                            core_smiles: str = "",
                            ) -> Optional[Decomposition]:
    """Assemble a Decomposition from a nominated core atom-set.

    Used by Murcko/MacFrag where the core is a sub-structure of M rather
    than the result of a single bond cut. The remainder is split into
    connected R-groups; the linker atom of each is identified as the unique
    R-group atom with at least one core neighbour.
    """
    core_set: Set[int] = set(core_atoms)
    if not core_set:
        return None
    if ratio is not None and len(core_set) < ratio * mol.GetNumHeavyAtoms():
        return None

    rest = set(range(mol.GetNumAtoms())) - core_set
    if not rest:                                  # nothing to detach
        return None

    rgroup_infos: List[RGroupInfo] = []
    for comp in _connected_components(mol, rest):
        linker_core = linker_rg = None
        for ai in comp:
            for nbr in mol.GetAtomWithIdx(ai).GetNeighbors():
                if nbr.GetIdx() in core_set:
                    bond = mol.GetBondBetweenAtoms(ai, nbr.GetIdx())
                    if bond is None or bond.IsInRing():
                        continue
                    linker_rg, linker_core = ai, nbr.GetIdx()
                    break
            if linker_rg is not None:
                break
        if linker_rg is None:
            return None                           # disconnected → no clean cut
        rgroup_infos.append(RGroupInfo(
            rgroup_atoms=tuple(sorted(comp)),
            core_linker=linker_core,
            rgroup_linker=linker_rg,
        ))

    if not rgroup_infos:
        return None

    return Decomposition(
        core_smiles=core_smiles,
        core_atoms=tuple(sorted(core_set)),
        rgroups=tuple(rgroup_infos),
    )


def dedup_by_core(decomps: Sequence[Decomposition]) -> List[Decomposition]:
    """Keep at most one Decomposition per unique core atom-set."""
    seen: Set[Tuple[int, ...]] = set()
    out: List[Decomposition] = []
    for d in decomps:
        if d.core_atoms in seen:
            continue
        seen.add(d.core_atoms)
        out.append(d)
    return out


def is_safe_decomposition(mol: Chem.Mol, d: Decomposition) -> bool:
    """Reject decompositions that won't round-trip cleanly.

    Failure modes ruled out here:

    * **Ring-crossing cuts** — every (core_linker, rgroup_linker) bond must
      exist and must NOT be a ring bond. Cutting a ring bond drops
      aromaticity (and is chemically meaningless for "detach an R-group").
    * **Geminal joints** — two R-groups sharing the same ``core_linker``
      atom. ``detach_rgroups_multi`` rejects this explicitly; we drop the
      decomp upstream so callers see only good entries.
    """
    used_core_linkers: Set[int] = set()
    for rg in d.rgroups:
        if rg.core_linker in used_core_linkers:
            return False                          # geminal
        used_core_linkers.add(rg.core_linker)
        bond = mol.GetBondBetweenAtoms(rg.core_linker, rg.rgroup_linker)
        if bond is None or bond.IsInRing():
            return False                          # missing or ring-crossing
    return True


def filter_safe(mol: Chem.Mol,
                decomps: Sequence[Decomposition]) -> List[Decomposition]:
    """Keep only decompositions that pass :func:`is_safe_decomposition`."""
    return [d for d in decomps if is_safe_decomposition(mol, d)]
