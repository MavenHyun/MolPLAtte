"""Shared helpers for Type 2 (Fragment Partition) decomposers.

Every decomposer in this package follows the same pattern:

1. Identify a set of **cleavable bonds** in M according to its chemistry
   rule (BRICS, RECAP, r-BRICS, MacFrag, ring-aware, Synton, DigFrag).
2. Call :func:`bonds_to_partitions` to materialise one or more
   :class:`~molplatte_prep.fragments.data_types.FragmentPartition` objects.

The shared converter handles:

* ``max_cuts=None`` (default) — one *maximal* partition that cuts all
  cleavable bonds simultaneously.
* ``max_cuts=k`` (1 ≤ k < N_bonds) — emit one partition per k-subset of
  the cleavable bonds (i.e. ``C(N_bonds, k)`` partitions, intermediate
  granularity).
* ``max_cuts=0`` — emit one trivial whole-molecule partition (no cuts).

Filters:

* Any bond that is missing in M or sits inside a ring is silently
  skipped (cutting a ring bond doesn't disconnect anything by itself).
* Partitions where cutting the chosen bond-set doesn't actually
  disconnect M (single fragment after cut) are dropped.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem

from ..fragment_types import FragmentPartition
from ..fragment_graph_ops import partition_from_bonds


def _filter_cleavable(mol: Chem.Mol,
                      bonds: Sequence[Tuple[int, int]],
                      max_small_ring: int = 8,
                      ) -> List[Tuple[int, int]]:
    """Drop bonds missing in M or inside a SMALL ring; dedupe.

    Previously this dropped every ring bond, which silently discarded MacFrag's
    own macrocycle handling: its ``SSSRsize_filter`` (vendor/macfrag.py:162)
    deliberately permits cutting a ring bond that is *not* in any ring of size
    3..maxSR, so macrolides and macrocyclic glycosides can be opened. Refusing
    all ring bonds meant those molecules could only ever be cut at the
    periphery -- 2.2% of COCONUT contains a ring larger than 8 atoms.

    A single ring cut does not disconnect anything; two or more on the same ring
    do. Non-separating cuts are dropped downstream in
    ``fragment_graph_ops._build_partition_from_adj`` rather than here, because
    whether a cut separates depends on the whole cut set, not on the bond alone.
    """
    seen: set = set()
    out: List[Tuple[int, int]] = []
    for a, b in bonds:
        a, b = int(a), int(b)
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None:
            continue
        if bond.IsInRing() and any(
            bond.IsInRingSize(k) for k in range(3, max_small_ring + 1)
        ):
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def bonds_to_partitions(mol: Chem.Mol,
                        bonds: Sequence[Tuple[int, int]],
                        max_cuts: Optional[int] = None,
                        ) -> List[FragmentPartition]:
    """Convert a list of cleavable bonds to fragment partitions.

    Parameters
    ----------
    mol : Chem.Mol
    bonds : sequence of (a, b)
        Cleavable bonds. Missing-in-M or in-ring entries are silently
        dropped.
    max_cuts : int or None, default None
        * ``None``: emit ONE partition that cuts every (filtered)
          bond simultaneously — the maximal partition.
        * ``0``: emit a single whole-molecule partition (no cuts).
        * ``k`` (1 ≤ k ≤ N_bonds): emit one partition per k-subset of
          the (filtered) bond list.

    Returns
    -------
    list[FragmentPartition]
        Possibly empty if (i) ``bonds`` filtered to empty and
        ``max_cuts >= 1``, or (ii) every enumerated cut-set fails to
        disconnect M.
    """
    n = mol.GetNumAtoms()
    bonds = _filter_cleavable(mol, bonds)

    if max_cuts == 0:
        # Single whole-molecule partition.
        return [partition_from_bonds(mol, [])]

    if not bonds:
        return []

    if max_cuts is None:
        # Maximal cut: all bonds at once.
        try:
            p = partition_from_bonds(mol, bonds)
        except ValueError:
            return []
        if p.n_fragments < 2:
            return []
        return [p]

    out: List[FragmentPartition] = []
    seen_signatures: set = set()
    for combo in combinations(bonds, max_cuts):
        try:
            p = partition_from_bonds(mol, combo)
        except ValueError:
            continue
        if p.n_fragments < 2:
            continue
        # Dedup by partition signature (sorted tuple of frag-atom-tuples).
        sig = tuple(f.atoms_in_M for f in p.fragments)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        out.append(p)
    return out


def _non_ring_single_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
    """Every non-ring SINGLE bond as (a, b). Used by the ring-aware
    decomposer when no rule-based bond set is available."""
    out: List[Tuple[int, int]] = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            continue
        out.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return out
