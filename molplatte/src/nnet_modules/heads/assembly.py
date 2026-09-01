"""Assembly head — recovering the chemistry that masking destroyed.

What it is for
--------------
R-group retrieval answers *which* substituent belongs at a linker joint.  It
cannot answer *how to attach it*, because the joint is masked: the shared linker
atom's identity and the bond that forms between core and R-group were both
replaced by MASK sentinels.  That masking is deliberate — it is what frees
retrieval from having to match the original linker's atom type and bond order —
but it means a retrieved R-group cannot be bonded onto the core without guessing.

MolPLA fills the joint with "proper node and edge attributes" via an unpublished
rule-based procedure (Supplementary S6).  MolDAM learns it.  This head learns it,
and is the **core-decoration** objective: given a core template and a candidate
R-group, predict the chemistry of the bond that joins them.

Two branches, independently switchable:

``recovery`` (default **on**)
    Per-attribute classifiers over the pre-mask chemistry at each joint — the
    atom the mask replaced, and the cut bond that reforms.  This is the branch
    with no MolPLA analogue and the reason to build the head at all.

``coupling`` (default **off**)
    InfoNCE pairing each core-side joint with its R-group clone.  Off by default
    because MolPLAtte's ``linker_contrastive`` loss already couples joints, in
    MolPLA's formulation (G-side against P+R summed) rather than MolDAM's
    (P-side against R-side).  Enable it to ablate the two against each other;
    running both is a redundant third linker-level objective.

One target per joint, not two
-----------------------------
MolDAM clones the cut atom on *both* sides and recovers each separately.  MolPLA's
shared-linker convention means the core joint and the R-group clone are the *same
atom*, so ``linker_metas`` carries one ``atom_features`` dict per joint and this
head predicts one target.  Verified: the P-side and R-side dicts are identical.

Fusion
------
The joint is seen twice — once masked in ``P``, once masked in ``R`` — and the two
embeddings must be combined before classification.  MolDAM uses ``0.5 * (core +
rgroup)`` and its own design doc flags that as halving the signal for asymmetric
joints.  ``fusion`` selects ``mean`` (MolDAM's), ``concat`` (default; a learned
projection over both) or ``gate`` (a learned convex combination).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..components.utils import FEAT2DIM_EDGE, FEAT2DIM_NODE, _make_norm

# Cardinalities are DERIVED from the featurisation vocabularies rather than
# restated, so they cannot drift from them. `nnet_modules` deliberately does not
# import `data_modules` (MolDAM's layering); the collate side owns the mirror of
# this policy in `data_modules/assembly_targets.py`, and the two are kept in
# agreement by both deriving from the same source of truth.
#
# chiral_tag and bond_dir are excluded, not merely defaulted off:
#   chiral_tag  CHI_TETRAHEDRAL_CW/CCW is defined relative to neighbour order,
#               and a joint's neighbour order changes by construction when the
#               R-group detaches. Ill-posed. `chirality_specified` replaces it.
#   bond_dir    encodes how E/Z was WRITTEN during traversal, not what it is;
#               bond_stereo carries the real thing and is recoverable.
_EXCLUDED_NODE = {"chiral_tag"}
_EXCLUDED_EDGE = {"bond_dir"}

NODE_ATTR_DIMS: Dict[str, int] = {
    **{k: v for k, v in FEAT2DIM_NODE.items() if k not in _EXCLUDED_NODE},
    "chirality_specified": 2,
}
EDGE_ATTR_DIMS: Dict[str, int] = {
    k: v for k, v in FEAT2DIM_EDGE.items() if k not in _EXCLUDED_EDGE
}

#: Defaults recover the atom identity and the bond that forms -- the minimal set
#: needed to actually attach a retrieved R-group. The rest are opt-in.
DEFAULT_NODE_ATTRS: List[str] = ["atomic_num", "formal_charge", "total_num_hs"]
DEFAULT_EDGE_ATTRS: List[str] = ["bond_type"]

__all__ = ["AssemblyHead"]

_FUSIONS = ("mean", "concat", "gate")


class AssemblyHead(nn.Module):
    """Predict the pre-mask chemistry at each linker joint.

    Parameters
    ----------
    hidden_dim, dropout_rate, norm_method
        Inherited from the model unless overridden.
    node_attrs, edge_attrs
        Which attributes to recover.  ``None`` uses the defaults (atom identity
        plus bond type — the minimal set needed to actually attach an R-group);
        pass explicit lists to widen or narrow.  Unknown names raise.
    fusion
        ``"concat"`` (default), ``"mean"`` or ``"gate"`` — see the module
        docstring.
    coupling
        Build the coupling projections.  Off by default; ``linker_contrastive``
        already covers this.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        hidden_dim: int = kwargs["hidden_dim"]
        dropout_rate: float = kwargs.get("dropout_rate", 0.0)
        norm_method: Optional[str] = kwargs.get("norm_method", "LayerNorm")

        node_attrs = kwargs.get("node_attrs") or list(DEFAULT_NODE_ATTRS)
        edge_attrs = kwargs.get("edge_attrs") or list(DEFAULT_EDGE_ATTRS)
        bad = [a for a in node_attrs if a not in NODE_ATTR_DIMS]
        bad += [a for a in edge_attrs if a not in EDGE_ATTR_DIMS]
        if bad:
            raise ValueError(
                f"unrecoverable attribute(s) {bad}; available: "
                f"{list(NODE_ATTR_DIMS)} (node) / {list(EDGE_ATTR_DIMS)} (edge). "
                f"chiral_tag and bond_dir are deliberately absent -- see "
                f"data_modules/assembly_targets.py"
            )

        self.node_attrs: List[str] = list(node_attrs)
        self.edge_attrs: List[str] = list(edge_attrs)
        self.fusion: str = kwargs.get("fusion", "concat")
        if self.fusion not in _FUSIONS:
            raise ValueError(f"unknown fusion {self.fusion!r}; available: {_FUSIONS}")
        self.coupling: bool = bool(kwargs.get("coupling", False))

        fuse_in = 2 * hidden_dim if self.fusion == "concat" else hidden_dim
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden_dim),
            _make_norm(norm_method, hidden_dim, graph_aware=False),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
        )
        if self.fusion == "gate":
            # Scalar-per-dimension convex mix, so an asymmetric joint can weight
            # the core side and the R-group side differently.
            self.gate = nn.Linear(2 * hidden_dim, hidden_dim)

        self.node_heads = nn.ModuleDict(
            {a: nn.Linear(hidden_dim, NODE_ATTR_DIMS[a]) for a in self.node_attrs}
        )
        # The cut bond is predicted from the fused joint too: it is the bond
        # between the two sides, so both endpoints are already represented.
        self.edge_heads = nn.ModuleDict(
            {a: nn.Linear(hidden_dim, EDGE_ATTR_DIMS[a]) for a in self.edge_attrs}
        )

        if self.coupling:
            self.proj_core = nn.Linear(hidden_dim, hidden_dim)
            self.proj_rgroup = nn.Linear(hidden_dim, hidden_dim)

    def _fuse(self, core: torch.Tensor, rgroup: torch.Tensor) -> torch.Tensor:
        if self.fusion == "mean":
            return self.fuse(0.5 * (core + rgroup))
        both = torch.cat([core, rgroup], dim=-1)
        if self.fusion == "gate":
            alpha = torch.sigmoid(self.gate(both))
            return self.fuse(alpha * core + (1.0 - alpha) * rgroup)
        return self.fuse(both)

    def forward(self, batch: Dict) -> Dict:
        """Reads ``joint_P_idx`` / ``joint_R_idx``; writes prediction logits."""
        H = batch["W"].node_embeddings
        core_joint = H[batch["joint_P_idx"]]
        rgroup_joint = H[batch["joint_R_idx"]]

        if core_joint.shape[0] == 0:
            # An empty batch of joints is legitimate (every R-group unpairable);
            # emit nothing and let the loss skip rather than crash on empty CE.
            batch["assembly_atom_pred"] = {}
            batch["assembly_bond_pred"] = {}
            return batch

        fused = self._fuse(core_joint, rgroup_joint)
        batch["assembly_atom_pred"] = {a: h(fused) for a, h in self.node_heads.items()}
        batch["assembly_bond_pred"] = {a: h(fused) for a, h in self.edge_heads.items()}

        if self.coupling:
            batch["assembly_core_embedding"] = F.normalize(
                self.proj_core(core_joint), dim=-1
            )
            batch["assembly_rgroup_embedding"] = F.normalize(
                self.proj_rgroup(rgroup_joint), dim=-1
            )
        return batch
