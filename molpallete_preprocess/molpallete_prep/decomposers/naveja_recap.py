"""Naveja-filtered RECAP decomposer (anchored paradigm, MolPLA-paper default).

Algorithm:

1. RDKit's RECAP recursion enumerates fragments of M.
2. Keep only fragments whose heavy-atom count is at least
   ``ratio * nHA(M)`` (Naveja 2019 core-size filter, default 2/3).
3. (Optional) Ring-aware fallback: also enumerate every non-ring single
   bond and apply the same threshold. Recovers cuts that RECAP refuses on
   fused-aromatic systems (e.g. caffeine's N--CH3).

Each surviving putative core defines a Decomposition: the core atom-set
is the kept fragment; R-groups are the remaining connected components of
M-core, each carrying its (core_linker, rgroup_linker) joint atoms.

The output passes through :func:`filter_safe` to drop any ring-crossing
cuts or geminal-joint configurations that would break round-trip
re-assembly.
"""
from __future__ import annotations

from typing import List

from rdkit import Chem

from ..decompose import (
    Decomposition,
    find_putative_cores,
    find_ring_substituent_cores,
)
from .anchored_common import filter_safe


def decompose_naveja_recap(mol: Chem.Mol,
                           ratio: float = 2.0 / 3.0,
                           include_ring: bool = True,
                           max_cores: int = 6,
                           ) -> List[Decomposition]:
    """Naveja putative cores: cut every RECAP bond, then take connected
    fragment-subtrees above ``ratio * n_atoms`` as cores.

    REWRITTEN. The previous implementation used RECAP *children* as cores
    (``Recap.RecapDecompose(...).GetAllChildren()``), and a child is a single
    connected fragment, so ``M - core`` was one pendant group almost always. That
    produced 1.08 R-groups per decomposition and only 6.7-9.5% of decompositions
    with k >= 2, which makes MolPLA's islinked subset space (``2^k - 1``) and the
    core-decoration objective nearly vacuous.

    MolPLA's released dataset shows that is not what its decomposition does. From
    ``molpla-datasets.tar.gz``, DrugBank split, ids ``{mol}-CORE-{core}-{islinked}``:

        k = 1   47.3%
        k >= 2  52.7%      cores/molecule 3.15

    -- majority multi-R-group, against 9.5% here. So the single-R-group behaviour
    was an artefact of the child-as-core construction, not a property of Naveja's
    method. (MolDAM_prep's own docstring had already retracted the claim that
    ratio=2/3 "reproduces MolPLA-paper behaviour"; this measures why.)

    The fix reuses the machinery that already produces MolPLA-shaped cores for
    the multi-cut methods: cut ALL RECAP bonds into a flat partition, then take
    connected subtrees of the resulting fragment tree as cores, with every
    boundary fragment-component becoming an R-group. ``ratio`` keeps its Naveja
    meaning -- a core must hold at least that fraction of the molecule's atoms.

    ``include_ring`` additionally admits ring-substituent cores, as before.
    """
    from ..anchored_from_partition import partitions_to_decompositions
    from .recap import decompose_recap

    parts = decompose_recap(mol)
    cands: List[Decomposition] = []
    if parts:
        # Uncapped here: the cap is applied once, globally, after the
        # ring-substituent cores are merged in. Capping before the merge let the
        # ring cores -- which are single-R-group by construction -- displace
        # multi-R-group candidates and pushed k>=2 back down from 46.5% to 20.1%.
        cands = partitions_to_decompositions(
            parts, mol.GetNumAtoms(), ratio=ratio, max_cores=1 << 30,
        )
    if include_ring and not cands:
        # FALLBACK ONLY. Ring-substituent cores are single-R-group by
        # construction, so mixing them in wholesale dilutes k: measured k>=2
        # falls 60.8% -> 36.7% when they always participate. Used only when the
        # RECAP fragment tree yields no core, they rescue the 26% of molecules
        # that would otherwise be dropped without costing k on the rest.
        cands.extend(find_ring_substituent_cores(mol, ratio))

    by_core = {}
    for d in cands:
        prev = by_core.get(d.core_atoms)
        if prev is None or len(d.rgroups) > len(prev.rgroups):
            by_core[d.core_atoms] = d
    ranked = sorted(
        by_core.values(),
        key=lambda d: (-len(d.rgroups), -len(d.core_atoms), d.core_atoms),
    )
    # MolPLA excluded molecules above 10 cores (its 99th percentile was 11) and
    # reports ~3-4 cores per molecule; ranking multi-R-group cores first means the
    # cap keeps the informative ones.
    return filter_safe(mol, ranked[:max_cores])
