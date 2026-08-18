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
                            ) -> List[Decomposition]:
    """RECAP children + Naveja core-size filter + ring-aware fallback.

    Parameters
    ----------
    ratio
        Heavy-atom share of M required of a candidate core. ``2/3`` is the
        long-standing default here; ``2/5`` and ``1/6`` are the more
        permissive Naveja settings.

        Measured on 3,000 fresh ZINC molecules (mean 23.6 heavy atoms), one
        decomposition sampled per molecule as the dataset draws it:

            ratio   dec/mol   core nHA   rgroup nHA   >=2 R-groups
            2/3        6.01       20.8          2.7          0.23%
            3/5        7.03       20.0          3.6          0.30%
            1/2        8.88       18.5          5.1          0.83%
            2/5        9.68       17.8          5.7          1.93%
            1/6          --       16.8           --          6.90%

        Read the last column before choosing: **this decomposer is
        single-R-group at every practical setting.** On the full 708,300-record
        ZINC-1pct corpus at 2/3, 99.4172% of records have exactly one R-group
        and only 0.5828% have two or more (max 4). Even 2/5 leaves 98% single.

        The cause is structural, not a threshold effect: ``find_putative_cores``
        takes RECAP *children* as candidate cores, and a child is one connected
        fragment, so ``M - core`` is usually one pendant group no matter how
        small the core is allowed to be. If genuinely multi-R-group
        decompositions are needed, lower ``ratio`` will not deliver them -- use
        a multi-cut decomposer (the fragments paradigm's ``max_cuts``).

        NOTE: the claim that 2/3 "reproduces MolPLA-paper behaviour", carried in
        this docstring historically, could NOT be verified -- the public MolPLA
        repository ships its dataloaders but not its decomposition step, and no
        ratio appears anywhere in it.
    include_ring
        If True (default), also apply the ring-aware fallback to recover
        non-ring single-bond cuts that RECAP doesn't reach.
    """
    decs = find_putative_cores(mol, ratio=ratio)
    if include_ring:
        extra = find_ring_substituent_cores(mol, ratio=ratio)
        seen = {d.core_atoms for d in decs}
        for d in extra:
            if d.core_atoms not in seen:
                decs.append(d)
                seen.add(d.core_atoms)
    return filter_safe(mol, decs)
