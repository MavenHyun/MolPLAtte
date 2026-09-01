"""Which pre-mask attributes the assembly head may recover, and why.

Masking a linker joint destroys the chemistry at the cut point -- deliberately,
since that is what frees R-group retrieval from having to match the original
linker's atom type and bond order.  The cost is that a retrieved R-group cannot
actually be bonded onto the core: the joint atom's identity and the bond that
forms are both unknown.  Recovery predicts them back.

Two attributes are excluded from the recoverable set, for the same reason they
were removed from the WL hash (see ``molplatte_prep/graph_hash.py``):

``chiral_tag``
    ``CHI_TETRAHEDRAL_CW/CCW`` is defined relative to the atom's neighbour order,
    and the joint atom's neighbour order changes by construction when the R-group
    is detached.  Predicting it is ill-posed.  ``chirality_specified`` -- a
    boolean -- is offered instead: order-independent, and it still says whether
    the joint is a stereocentre.
``bond_dir``
    ``ENDUPRIGHT``/``ENDDOWNRIGHT`` encode how E/Z was *written* during SMILES
    traversal.  Predicting it means fitting a serialisation convention.
    ``bond_stereo`` carries the actual E/Z and is recoverable.

``is_conjugated`` **is** recoverable here even though it was dropped from the
hash.  The two uses differ: in a vocabulary key it was parent-context leaking
into an identity, but as a recovery target, reconstructing the parent's
conjugation is exactly the intended prediction.
"""

from __future__ import annotations

from typing import Dict, List

__all__ = [
    "RECOVERABLE_NODE_ATTRS",
    "RECOVERABLE_EDGE_ATTRS",
    "DEFAULT_NODE_ATTRS",
    "DEFAULT_EDGE_ATTRS",
    "NODE_ATTR_DIMS",
    "EDGE_ATTR_DIMS",
    "derive_node_target",
]

#: Cardinality of each recoverable target. ``chirality_specified`` is binary; the
#: rest match the featurisation vocabularies (MASK indices are never targets --
#: the point is to predict the value the mask replaced).
NODE_ATTR_DIMS: Dict[str, int] = {
    "atomic_num": 128,
    "formal_charge": 11,
    "chirality_specified": 2,
    "hybridization": 9,
    "total_num_hs": 9,
    "is_aromatic": 2,
    "is_in_ring": 2,
}
EDGE_ATTR_DIMS: Dict[str, int] = {
    "bond_type": 22,
    "edge_is_aromatic": 2,
    "is_conjugated": 2,
    "bond_stereo": 8,
    "edge_is_in_ring": 2,
}

RECOVERABLE_NODE_ATTRS: List[str] = list(NODE_ATTR_DIMS)
RECOVERABLE_EDGE_ATTRS: List[str] = list(EDGE_ATTR_DIMS)

#: Defaults recover the atom identity and the bond that forms -- the minimal set
#: needed to actually attach a retrieved R-group. The rest are opt-in.
DEFAULT_NODE_ATTRS: List[str] = ["atomic_num", "formal_charge", "total_num_hs"]
DEFAULT_EDGE_ATTRS: List[str] = ["bond_type"]


def derive_node_target(attr: str, atom_features: Dict[str, int]) -> int:
    """Target class for *attr* from a ``linker_metas`` ``atom_features`` dict.

    ``chirality_specified`` is derived from the raw ``chiral_tag`` -- index 0 is
    ``CHI_UNSPECIFIED``, so anything else means the joint carries specified
    stereochemistry.
    """
    if attr == "chirality_specified":
        return int(int(atom_features.get("chiral_tag", 0)) != 0)
    return int(atom_features.get(attr, 0))
