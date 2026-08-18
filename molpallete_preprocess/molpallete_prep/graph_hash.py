"""Weisfeiler-Lehman sub-graph hash used as vocabulary key.

Lifted out of :mod:`molpallete_prep.anchored.graph_ops` so this ``data_modules``
package is self-contained (no need to import the full ``molpallete_prep`` package
to use the datasets).
"""
from __future__ import annotations

from torch_geometric.data import Data


#: Bumped whenever the label set changes. Stored in corpus and vocabulary
#: provenance so a corpus hashed under one version cannot be silently paired
#: with a library built under another -- the stored ``rgroup_hashes``, the
#: library keys and the loss's multi-positive grouping must all agree, and a
#: mismatch would degrade retrieval quietly rather than fail.
#:
#: v1: (atomic_num, formal_charge, chiral_tag, hybridization, num_explicit_hs,
#:      is_aromatic, is_linker) / (bond_type, edge_is_aromatic, is_conjugated,
#:      bond_dir, bond_stereo, edge_is_linker)
#: v2: chiral_tag -> chirality_specified; dropped is_conjugated and bond_dir.
HASH_VERSION = 2


def subgraph_hash(data: Data, iterations: int = 3, digest_size: int = 16) -> str:
    """Isomorphism-invariant Weisfeiler-Lehman hash of a masked-linker
    sub-graph. Use this as the vocabulary key when SMILES isn't an
    option (masked atoms / bonds are not real chemical entities).

    Labels used:
      - Node: (atomic_num, formal_charge, chirality_specified,
        hybridization, num_explicit_hs, is_aromatic, is_linker).
      - Edge: (bond_type, edge_is_aromatic, bond_stereo, edge_is_linker).

    Three attributes are deliberately EXCLUDED because they are not
    properties of the isolated sub-graph. Including them splits
    chemically identical R-groups across several library rows, which
    inflates the row count, divides an R-group's frequency mass, and
    depresses Hit@K by letting a query's mass land on an identical
    sibling that scores as a miss. Measured on a 60,773-row library:
    6,235 SMILES labels were split across multiple hashes, and these
    three were the sole cause for 3,208 of them.

    ``chiral_tag`` (sole cause of 2,562 splits)
        ``CHI_TETRAHEDRAL_CW/CCW`` is defined relative to the atom's
        NEIGHBOUR ORDER in the molecule it was computed on.
        ``detach_rgroups_multi`` drops the core-side neighbour and
        appends a masked clone at the end of the atom list, so the same
        3D configuration serialises as CW in one decomposition and CCW
        in another. Replaced by ``chirality_specified`` -- a boolean that
        is order-independent and still separates stereocentres from
        achiral atoms.

        Cost: R and S enantiomers now share a library row. Recovering
        that needs order-independent CIP codes, which require a
        sanitised Mol per call and are far too slow here (this runs
        ~8.7M times per corpus). Bond stereochemistry, which carries
        most of the flavour-relevant isomerism (E/Z, e.g. (Z)- vs
        (E)-3-hexenol), is unaffected -- see ``bond_dir`` below.

    ``bond_dir`` (sole cause of 485 splits)
        ``ENDUPRIGHT``/``ENDDOWNRIGHT`` encode how E/Z was *written*
        during SMILES traversal, not what it is. ``bond_stereo``
        (``STEREOE``/``STEREOZ``) is the canonical form and is retained,
        so no real stereochemistry is lost.

    ``is_conjugated`` (sole cause of 161 splits)
        Computed over the PARENT molecule's conjugated system, so a
        detached fragment carries a flag describing atoms that are no
        longer present -- two identical fragments differ according to
        what they used to be attached to.

    ``linker_id`` values are likewise NOT part of the hash. Two
    structurally identical sub-graphs that carry different joint IDs
    hash to the same key.

    Note that ``hybridization`` and ``num_explicit_hs`` ARE retained
    despite also splitting same-SMILES groups: both are intrinsic to the
    atom, and the splitting there reflects the SMILES *label* being
    lossy rather than the hash being wrong.

    Returns
    -------
    str
        ``2 * digest_size`` hex characters.
    """
    import networkx as nx

    G = nx.Graph()
    for i in range(data.num_nodes):
        # Boolean, not the raw tag: CHI_UNSPECIFIED is index 0, so anything
        # else means "this atom carries specified stereochemistry" without
        # committing to an order-dependent CW/CCW value.
        node_label = (
            int(data.atomic_num[i].item()),
            int(data.formal_charge[i].item()),
            int(int(data.chiral_tag[i].item()) != 0),
            int(data.hybridization[i].item()),
            int(data.num_explicit_hs[i].item()),
            int(data.is_aromatic[i].item()),
            int(data.is_linker[i].item()),
        )
        G.add_node(i, label=repr(node_label))
    seen: set = set()
    for e in range(data.edge_index.size(1)):
        u = int(data.edge_index[0, e].item())
        v = int(data.edge_index[1, e].item())
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        edge_label = (
            int(data.bond_type[e].item()),
            int(data.edge_is_aromatic[e].item()),
            int(data.bond_stereo[e].item()),
            int(data.edge_is_linker[e].item()),
        )
        G.add_edge(u, v, label=repr(edge_label))
    return nx.weisfeiler_lehman_graph_hash(
        G, edge_attr="label", node_attr="label",
        iterations=iterations, digest_size=digest_size,
    )
