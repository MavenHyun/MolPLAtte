"""MolPLAtte preprocessing — flavor-compound corpora for MolPLA-style pretraining.

Reads FlavorDB / COCONUT structures, decomposes each molecule into anchored
cores + R-groups, and writes one record per molecule containing the intact graph
plus every decomposition's bookkeeping.  Training-time sampling of the
``(decomposition, islinked)`` pair happens in the training repo.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .anchored_from_partition import (
    partition_to_decompositions,
    partitions_to_decompositions,
)
from .decompose import Decomposition, RGroupInfo, wash
from .decomposers import (
    family_of,
    get_anchored_decomposer,
    get_fragment_decomposer,
    list_methods,
)
from .graph_hash import subgraph_hash
from .mol_features import EDGE_ATTRS, NODE_ATTRS, MolPLAtteData, mol_to_pyg
from .molpla_instance import (
    MolPlaInstance,
    build_instance,
    decomposition_record,
    enumerate_islinked,
    sample_islinked,
)

__all__ = [
    "Decomposition",
    "EDGE_ATTRS",
    "MolPLAtteData",
    "MolPlaInstance",
    "NODE_ATTRS",
    "RGroupInfo",
    "build_instance",
    "decomposition_record",
    "enumerate_islinked",
    "family_of",
    "get_anchored_decomposer",
    "get_fragment_decomposer",
    "list_methods",
    "mol_to_pyg",
    "partition_to_decompositions",
    "partitions_to_decompositions",
    "sample_islinked",
    "subgraph_hash",
    "wash",
]
