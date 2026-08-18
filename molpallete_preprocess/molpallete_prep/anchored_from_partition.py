"""Re-frame a flat multi-cut ``FragmentPartition`` into MolPLA anchored stars.

MolPLA's formulation needs one **core** plus *k* **R-groups**, each R-group joined
to the core by exactly one shared linker atom.  Multi-cut decomposers
(``macfrag``, ``synton``) instead return a flat partition of the molecule into
K mutually disjoint fragments plus the bonds that were cut.  This module bridges
the two.

The bridge
----------
A partition induces a **fragment tree**: vertices are fragments, edges are cut
bonds.  (It is a tree, not a general graph, because every cut bond is acyclic —
the decomposers' ``_filter_cleavable`` drops ring bonds, so cutting can never
create a cycle among fragments.)

Pick any *connected* vertex subset ``C`` of that tree as the core.  Removing ``C``
leaves one connected component per edge crossing the ``C`` boundary, and each such
component touches ``C`` through exactly one cut bond.  That is precisely a star:

    core = union of C's atoms
    R-group i = union of the atoms of component i, attached at cut bond i

Enumerating connected subsets whose atom count clears ``ratio * n_atoms`` reproduces
MolPLA's "single molecule - multiple putative cores" property, with ``ratio``
playing the same role as in Naveja's putative-core criterion.

Measured on 400 FlavorDB molecules (heavy atoms in [5, 50]) via ``macfrag``:

===========  ==========  ===========  ==========  ==========  =========
ratio        no-decomp   cores/mol    R-grp/core  k >= 2      core nHA
===========  ==========  ===========  ==========  ==========  =========
0.33         12.2%       79.30        3.26        89.7%       20.7
0.50         12.2%       64.45        3.34        89.7%       22.5
0.67         16.8%       45.65        3.23        88.6%       24.7
===========  ==========  ===========  ==========  ==========  =========

For reference MolPLA reports ~4.04 cores per molecule and ~20.8-heavy-atom cores
on GEOM, so ``ratio=0.5`` matches its core size while producing far more
candidates than are wanted.  ``max_cores`` prunes to a MolPLA-like budget; MolPLA
itself dropped molecules above 10 cores (its 99th percentile was 11).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, FrozenSet, List, Sequence, Tuple

from .decompose import Decomposition, RGroupInfo
from .fragment_types import FragmentPartition

__all__ = ["partition_to_decompositions", "partitions_to_decompositions"]

#: Hard ceiling on enumerated connected subsets, so a pathological partition with
#: many fragments cannot stall a worker. K fragments admit up to 2**K subsets.
_MAX_SUBSETS = 4096


def _fragment_adjacency(partition: FragmentPartition) -> Dict[int, List[tuple]]:
    """``frag_idx -> [(neighbour_frag_idx, CutBond), ...]``."""
    adj: Dict[int, List[tuple]] = defaultdict(list)
    for cb in partition.cut_bonds:
        adj[cb.u_frag].append((cb.v_frag, cb))
        adj[cb.v_frag].append((cb.u_frag, cb))
    return adj


def _connected_subsets(
    adj: Dict[int, List[tuple]], n_frags: int, max_subsets: int = _MAX_SUBSETS
) -> List[FrozenSet[int]]:
    """Every connected vertex subset of the fragment tree, each yielded once.

    Canonicalised by only growing a subset with vertices greater than the seed,
    which makes each connected subset reachable from exactly one seed.
    """
    out: List[FrozenSet[int]] = []
    seen: set = set()
    for seed in range(n_frags):
        stack: List[FrozenSet[int]] = [frozenset((seed,))]
        while stack:
            subset = stack.pop()
            if subset in seen:
                continue
            seen.add(subset)
            out.append(subset)
            if len(out) >= max_subsets:
                return out
            for v in subset:
                for nbr, _cb in adj[v]:
                    if nbr not in subset and nbr > seed:
                        stack.append(subset | {nbr})
    return out


def _component_outside(
    adj: Dict[int, List[tuple]], start: int, core: FrozenSet[int]
) -> set:
    """Fragments reachable from *start* without entering *core*."""
    seen = {start}
    queue = deque([start])
    while queue:
        f = queue.popleft()
        for nbr, _cb in adj[f]:
            if nbr not in core and nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)
    return seen


def partition_to_decompositions(
    partition: FragmentPartition,
    n_atoms: int,
    ratio: float = 0.5,
    min_core_atoms: int = 3,
    min_rgroup_atoms: int = 1,
    max_rgroups: int = 8,
    max_subsets: int = _MAX_SUBSETS,
) -> List[Decomposition]:
    """Enumerate anchored decompositions induced by *partition*.

    Parameters
    ----------
    partition
        A flat fragment partition from ``macfrag`` / ``synton``.
    n_atoms
        Atom count of the parent molecule; ``ratio`` is taken against this.
    ratio
        A candidate core must hold at least ``ratio * n_atoms`` atoms.
    min_core_atoms, min_rgroup_atoms
        Absolute floors, applied on top of ``ratio``.
    max_rgroups
        Reject cores with more than this many R-groups.  The training-time
        ``islinked`` subset space is ``2**k - 1``, so this bounds sampling cost.
    max_subsets
        Ceiling on enumerated connected subsets (runaway guard).

    Returns
    -------
    list of Decomposition
        Possibly empty.  ``core_smiles`` is left ``""`` — computing it costs a
        SMILES round-trip per candidate and nothing downstream needs it.

    Notes
    -----
    Decompositions with a **geminal joint** (two R-groups sharing one core linker
    atom) are rejected: ``graph_ops.detach_rgroups_multi`` represents at most one
    joint per template atom, and MolPLA's ``is_linker`` marker is likewise one
    bit per atom.
    """
    n_frags = partition.n_fragments
    if n_frags < 2:
        return []

    adj = _fragment_adjacency(partition)
    frag_atoms = [set(f.atoms_in_M) for f in partition.fragments]
    threshold = max(ratio * n_atoms, float(min_core_atoms))

    results: List[Decomposition] = []
    for core in _connected_subsets(adj, n_frags, max_subsets):
        if len(core) == n_frags:
            continue  # a core spanning every fragment leaves no R-group
        core_atoms = set().union(*(frag_atoms[i] for i in core))
        if len(core_atoms) < threshold:
            continue

        rgroups: List[RGroupInfo] = []
        ok = True
        for c in core:
            for nbr, cb in adj[c]:
                if nbr in core:
                    continue
                comp = _component_outside(adj, nbr, core)
                atoms = tuple(sorted(a for f in comp for a in frag_atoms[f]))
                if len(atoms) < min_rgroup_atoms:
                    ok = False
                    break
                # Orient the cut bond: which endpoint sits on the core side?
                if cb.u_frag == c:
                    core_linker, rgroup_linker = cb.u_atom_in_M, cb.v_atom_in_M
                else:
                    core_linker, rgroup_linker = cb.v_atom_in_M, cb.u_atom_in_M
                if rgroup_linker not in atoms or core_linker not in core_atoms:
                    ok = False
                    break
                rgroups.append(RGroupInfo(atoms, core_linker, rgroup_linker))
            if not ok:
                break

        if not ok or not rgroups or len(rgroups) > max_rgroups:
            continue
        if len({r.core_linker for r in rgroups}) != len(rgroups):
            continue  # geminal joint

        results.append(
            Decomposition(
                core_smiles="",
                core_atoms=tuple(sorted(core_atoms)),
                rgroups=tuple(rgroups),
            )
        )
    return results


def partitions_to_decompositions(
    partitions: Sequence[FragmentPartition],
    n_atoms: int,
    max_cores: int = 10,
    **kwargs,
) -> List[Decomposition]:
    """Run :func:`partition_to_decompositions` over *partitions* and prune.

    Deduplicates on ``core_atoms`` (the same core can be reachable from several
    partitions), then keeps the ``max_cores`` best by *more R-groups first, larger
    core second*.  Richer decoration is the point of the core-decoration
    objective, so R-group count leads the ranking.
    """
    by_core: Dict[Tuple[int, ...], Decomposition] = {}
    for part in partitions:
        for dec in partition_to_decompositions(part, n_atoms, **kwargs):
            existing = by_core.get(dec.core_atoms)
            if existing is None or len(dec.rgroups) > len(existing.rgroups):
                by_core[dec.core_atoms] = dec

    ranked = sorted(
        by_core.values(),
        key=lambda d: (-len(d.rgroups), -len(d.core_atoms), d.core_atoms),
    )
    return ranked[:max_cores]
