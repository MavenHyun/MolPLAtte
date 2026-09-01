"""Build MolPLA data instances (G, P, R, islinked) from an anchored decomposition.

Notation
--------
MolPLA's paper and its reference implementation use different letters for the
same objects.  MolPallete standardises on the **code's** letters:

======  ==================  =========================================================
Code    Paper (btae256)     Content
======  ==================  =========================================================
``G``   :math:`G_M`         the intact molecule, all linker flags cleared
``P``   :math:`G_{Q}`       core + the R-groups that stay attached; one *masked*
                            linker atom per detached R-group
``R``   :math:`G_{R}`       each detached R-group, one graph each, one masked linker
``Q``   :math:`G_{D}`       ``P`` u ``R`` -- formed at collate time, not here
======  ==================  =========================================================

The paper's :math:`G_Q` ("query template") is the code's ``P``.  Do not mix them.

``islinked`` is a tuple of booleans, one per R-group of the decomposition:
``True`` means the R-group stays attached (it becomes part of ``P``), ``False``
means it is detached and becomes a retrieval target in ``R``.  The paper has no
name for this; it indexes non-empty subsets *k* of the R-group set.  At least one
R-group must be detached, so the subset space per decomposition is
:math:`2^k - 1`.

Sampling strategy
-----------------
Preprocessing stores the intact graph plus the **R-group bookkeeping** for every
decomposition -- not the detached graphs themselves.  Materialising all
:math:`2^k - 1` subsets on disk would multiply the corpus by ~9x at the measured
mean of 3.34 R-groups per core.  Instead the training-time dataset draws one
``(decomposition, islinked)`` pair per ``__getitem__`` and calls
:func:`build_instance` on the spot, which is the MolDAM sampling idiom applied to
MolPLA's subset space.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data

from .decompose import Decomposition, RGroupInfo
from .graph_hash import subgraph_hash
from .graph_ops import detach_rgroups_multi

__all__ = [
    "MolPlaInstance",
    "build_instance",
    "sample_islinked",
    "enumerate_islinked",
    "decomposition_record",
]


@dataclass
class MolPlaInstance:
    """One MolPLA training instance.

    Attributes
    ----------
    G
        The intact molecule graph, linker flags cleared.
    P
        Core plus still-attached R-groups, carrying one masked linker atom per
        detached R-group.
    R
        The detached R-group graphs, in decomposition order; each carries exactly
        one masked linker clone.
    islinked
        Per-R-group attachment flags of the parent decomposition.
    detached_indices
        Positions in the decomposition's R-group tuple that ``R`` corresponds to.
    joint_linker_ids
        The ``linker_id`` stamped on each joint, parallel to ``R``.  Both ``P``
        and the matching ``R`` graph carry this id, which is how the collate pairs
        a linker atom's two masked incarnations without relying on ordering.
    joint_G_atoms
        Parent-molecule atom index of each shared linker atom, parallel to ``R``.
        Because ``G`` *is* the parent molecule, this indexes straight into ``G`` --
        it is the third incarnation of the same atom and the anchor of the
        linker-node objective.
    R_hashes
        Weisfeiler-Lehman subgraph hash per detached R-group -- the retrieval
        vocabulary key, and the multi-positive grouping key for the contrastive
        loss.
    P_hash
        Subgraph hash of ``P``.
    instance_id
        ``"{mol_id}#{decomp_idx}-{islinked bitstring}"``, mirroring MolPLA's
        ``data_instance_id``.
    """

    G: Data
    P: Data
    R: List[Data]
    islinked: Tuple[bool, ...]
    detached_indices: Tuple[int, ...]
    joint_linker_ids: Tuple[int, ...]
    joint_G_atoms: Tuple[int, ...]
    R_hashes: List[str]
    P_hash: str
    instance_id: str


def enumerate_islinked(n_rgroups: int) -> List[Tuple[bool, ...]]:
    """Every attachment pattern with at least one R-group detached.

    Returns ``2**n_rgroups - 1`` tuples.  Use only for small *n*; the training
    dataset samples instead (see :func:`sample_islinked`).
    """
    out = []
    for mask in range(1 << n_rgroups):
        flags = tuple(bool(mask >> i & 1) for i in range(n_rgroups))
        if all(flags):
            continue  # at least one R-group must be detached
        out.append(flags)
    return out


def sample_islinked(
    n_rgroups: int, rng: Optional[random.Random] = None
) -> Tuple[bool, ...]:
    """Draw one attachment pattern uniformly from the ``2**n - 1`` valid ones.

    Rejection-samples the all-attached pattern, which has probability
    ``2**-n <= 0.5``, so this terminates in ~2 draws worst case (n = 1).
    """
    if n_rgroups < 1:
        raise ValueError(f"n_rgroups must be >= 1, got {n_rgroups}")
    r = rng or random
    while True:
        flags = tuple(r.random() < 0.5 for _ in range(n_rgroups))
        if not all(flags):
            return flags


def _clear_linker_flags(data: Data) -> Data:
    """Return a shallow copy of *data* with every linker flag cleared.

    MolPLA's ``G`` view is the *intact* molecule: ``G_data.is_linker &= False``
    and ``edge_is_linker &= False`` in the reference implementation, so the
    encoder sees no decomposition hint on the full-graph branch.
    """
    out = data.__class__()
    for key, value in data:
        out[key] = value
    out.is_linker = torch.zeros(data.num_nodes, dtype=torch.bool)
    out.edge_is_linker = torch.zeros(data.num_edges, dtype=torch.bool)
    return out


def build_instance(
    mol_data: Data,
    decomposition: Decomposition,
    islinked: Sequence[bool],
    mol_id: str = "",
    decomp_idx: int = 0,
    store_orig: bool = True,
    compute_hashes: bool = True,
) -> MolPlaInstance:
    """Materialise one MolPLA instance.

    Parameters
    ----------
    mol_data
        PyG graph of the intact molecule, as produced by
        :func:`molpallete_prep.mol_features.mol_to_pyg`.
    decomposition
        The anchored decomposition supplying core and R-group bookkeeping.
    islinked
        One flag per R-group; ``True`` keeps it attached.  At least one must be
        ``False``.
    store_orig
        Keep ``linker_metas`` (pre-mask atom/bond features at each joint).  These
        are the targets of the core-decoration auxiliary objective; drop them
        only if that objective is disabled.
    compute_hashes
        Compute WL subgraph hashes.  Cheap relative to the detach, and required
        for multi-positive contrastive masking.

    Raises
    ------
    ValueError
        If ``islinked`` has the wrong length or leaves nothing detached.
    """
    rgroups: Tuple[RGroupInfo, ...] = decomposition.rgroups
    if len(islinked) != len(rgroups):
        raise ValueError(
            f"islinked has {len(islinked)} flags but the decomposition has "
            f"{len(rgroups)} R-groups"
        )
    detached_indices = tuple(i for i, keep in enumerate(islinked) if not keep)
    if not detached_indices:
        raise ValueError("at least one R-group must be detached (islinked all True)")

    # detach_rgroups_multi punches the core-side atom in place and appends a
    # masked clone to each R-group -- exactly MolPLA's one-shared-linker-atom
    # convention.  The returned template is P: the core plus every R-group we did
    # NOT hand it, i.e. the still-attached ones.
    infos = [
        (rgroups[i].rgroup_atoms, rgroups[i].core_linker, rgroups[i].rgroup_linker)
        for i in detached_indices
    ]
    # Pass ids explicitly rather than leaning on the default 1..k numbering, so
    # the pairing the collate depends on is stated here, not inferred downstream.
    linker_ids = tuple(range(1, len(infos) + 1))
    P, R = detach_rgroups_multi(
        mol_data, infos, store_orig=store_orig, ids=list(linker_ids)
    )
    # The shared linker atom keeps its parent-molecule index, and G is the parent
    # molecule, so the core-side linker index indexes straight into G.
    joint_G_atoms = tuple(int(rgroups[i].core_linker) for i in detached_indices)

    G = _clear_linker_flags(mol_data)

    bits = "".join("1" if k else "0" for k in islinked)
    instance_id = f"{mol_id}#{decomp_idx}-{bits}"

    if compute_hashes:
        P_hash = subgraph_hash(P)
        R_hashes = [subgraph_hash(r) for r in R]
    else:
        P_hash, R_hashes = "", []

    return MolPlaInstance(
        G=G,
        P=P,
        R=R,
        islinked=tuple(bool(x) for x in islinked),
        detached_indices=detached_indices,
        joint_linker_ids=linker_ids,
        joint_G_atoms=joint_G_atoms,
        R_hashes=R_hashes,
        P_hash=P_hash,
        instance_id=instance_id,
    )


def decomposition_record(decomposition: Decomposition) -> Dict:
    """Serialise a decomposition to the plain-Python form stored on disk.

    Only bookkeeping is stored -- no graphs.  ``build_instance`` reconstructs the
    detached views at sampling time from ``original`` plus these indices.
    """
    return {
        "core_atoms": tuple(int(a) for a in decomposition.core_atoms),
        "core_smiles": decomposition.core_smiles,
        "n_rgroups": len(decomposition.rgroups),
        "rgroups": [
            {
                "rgroup_atoms": tuple(int(a) for a in rg.rgroup_atoms),
                "core_linker": int(rg.core_linker),
                "rgroup_linker": int(rg.rgroup_linker),
            }
            for rg in decomposition.rgroups
        ],
    }
