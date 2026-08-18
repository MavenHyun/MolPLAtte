"""Weisfeiler-Lehman sub-graph hash used as vocabulary key.

Lifted out of :mod:`molpallete_prep.anchored.graph_ops` so this ``data_modules``
package is self-contained (no need to import the full ``molpallete_prep`` package
to use the datasets).
"""
from __future__ import annotations

from torch_geometric.data import Data


def subgraph_hash(data: Data, iterations: int = 3, digest_size: int = 16) -> str:
    """Isomorphism-invariant Weisfeiler-Lehman hash of a masked-linker
    sub-graph. Use this as the vocabulary key when SMILES isn't an
    option (masked atoms / bonds are not real chemical entities).

    Labels used:
      - Node: (atomic_num, formal_charge, chiral_tag, hybridization,
        num_explicit_hs, is_aromatic, is_linker).
      - Edge: (bond_type, edge_is_aromatic, is_conjugated, bond_dir,
        bond_stereo, edge_is_linker).

    ``linker_id`` values are NOT part of the hash. Two structurally
    identical sub-graphs that carry different joint IDs hash to the
    same key.

    Returns
    -------
    str
        ``2 * digest_size`` hex characters.
    """
    import networkx as nx

    G = nx.Graph()
    for i in range(data.num_nodes):
        node_label = (
            int(data.atomic_num[i].item()),
            int(data.formal_charge[i].item()),
            int(data.chiral_tag[i].item()),
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
            int(data.is_conjugated[e].item()),
            int(data.bond_dir[e].item()),
            int(data.bond_stereo[e].item()),
            int(data.edge_is_linker[e].item()),
        )
        G.add_edge(u, v, label=repr(edge_label))
    return nx.weisfeiler_lehman_graph_hash(
        G, edge_attr="label", node_attr="label",
        iterations=iterations, digest_size=digest_size,
    )
