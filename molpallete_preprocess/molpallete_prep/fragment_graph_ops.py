"""Primitive graph operations for the Type 2 (Fragment Partition) paradigm.

This module mirrors :mod:`molpallete_prep.anchored.graph_ops` but for the
**flat partition** paradigm (no privileged "core"). The two operations
exposed are:

* :func:`detach_fragments` — given M (PyG ``Data``) and a
  :class:`~molpallete_prep.fragments.data_types.FragmentPartition`, return a
  list of PyG ``Data`` objects (one per fragment) each carrying
  masked-clone joints for every incident cut bond.
* :func:`attach_fragments` — inverse: glue a list of masked-clone
  fragments back into one molecule by pairing clones via
  ``linker_id``.

The :func:`subgraph_hash` helper is re-exported from
:mod:`molpallete_prep.anchored.graph_ops` so the same WL hash backs the
**one-vocab** fragments-paradigm vocabulary.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from rdkit import Chem
from torch_geometric.data import Data

from .graph_ops import (
    _append_atom,
    _append_bond,
    _atom_feature_snapshot,
    _default_bond_features,
    _edge_feature_snapshot,
    _ensure_molpla_attrs,
    _extract_subgraph,
    _find_edge_indices,
    mask_linker_atom,
    subgraph_hash,
)
from .mol_features import EDGE_ATTRS, MolPalleteData, NODE_ATTRS
from .fragment_types import CutBond, FragmentInfo, FragmentPartition, validate_partition


# ---------------------------------------------------------------------------
# detach
# ---------------------------------------------------------------------------

def detach_fragments(mol_data: Data,
                     partition: FragmentPartition,
                     store_orig: bool = True,
                     ) -> List[Data]:
    """Slice ``mol_data`` into k PyG fragments per ``partition``.

    For each cut bond ``C_k = (u, v)`` with ``u`` in fragment ``F_a`` and
    ``v`` in fragment ``F_b``:

    * ``F_a`` gets a fresh masked clone (cloned from ``v``'s atom
      features, then overwritten by MASK values) connected to ``F_a``'s
      ``u`` via a masked edge whose pre-mask features came from the
      original ``u--v`` bond.
    * ``F_b`` gets a fresh masked clone (cloned from ``u``'s atom
      features) connected to ``F_b``'s ``v`` similarly.

    Both clones share ``linker_id=k``; the pair is what
    :func:`attach_fragments` re-merges.

    Parameters
    ----------
    mol_data : Data
        The intact molecule M's PyG ``Data``. Not mutated.
    partition : FragmentPartition
        A validated partition of M.
    store_orig : bool, default True
        If True, snapshot the original cut bond's edge features into
        each clone's ``linker_metas[k]["cut_bond_features"]`` so attach
        can perfectly restore the bond. Set False for production
        preprocessing to save memory.

    Returns
    -------
    list[Data]
        One ``Data`` per fragment, indexed by fragment position in
        ``partition.fragments``. Each fragment carries:

        * standard MolPLA attrs (``atomic_num``, ``formal_charge``, …),
        * ``is_linker`` and ``edge_is_linker`` flags,
        * ``linker_id`` LongTensor (positive at clone positions),
        * ``linker_metas`` dict keyed by ``linker_id``.
    """
    validate_partition(partition, mol_data.num_nodes)

    # ---- snapshot per-cut bond/atom features BEFORE we touch anything -----
    cut_meta: Dict[int, Dict] = {}
    for c in partition.cut_bonds:
        es = _find_edge_indices(mol_data.edge_index, c.u_atom_in_M, c.v_atom_in_M)
        if not es:
            raise ValueError(
                f"cut bond {c}: no edge between atoms "
                f"{c.u_atom_in_M} and {c.v_atom_in_M} in M")
        cut_meta[c.linker_id] = {
            "u_atom_features": _atom_feature_snapshot(mol_data, c.u_atom_in_M),
            "v_atom_features": _atom_feature_snapshot(mol_data, c.v_atom_in_M),
            "cut_bond_features": _edge_feature_snapshot(mol_data, es[0]),
        }

    # ---- collect, per fragment, the list of clones to append -------------
    # Each entry: (linker_id, clone_source_atom_in_M, anchor_atom_in_M, is_u_side)
    # is_u_side = True if THIS fragment contains the u endpoint of the cut
    # (so the clone we append is cloned from v's atom features, anchored at u).
    per_frag_clones: List[List[Tuple[int, int, int, bool]]] = \
        [[] for _ in partition.fragments]
    for c in partition.cut_bonds:
        # In F_a (containing u): the clone is v's clone, anchored at u.
        per_frag_clones[c.u_frag].append(
            (c.linker_id, c.v_atom_in_M, c.u_atom_in_M, True))
        # In F_b (containing v): the clone is u's clone, anchored at v.
        per_frag_clones[c.v_frag].append(
            (c.linker_id, c.u_atom_in_M, c.v_atom_in_M, False))

    # ---- build each fragment ---------------------------------------------
    out: List[Data] = []
    for frag_idx, frag in enumerate(partition.fragments):
        frag_atoms_sorted = sorted(frag.atoms_in_M)
        sub, old_to_new, _ = _extract_subgraph(mol_data, frag_atoms_sorted)
        # _extract_subgraph already returns linker_id (zeros for fresh extract).
        if not hasattr(sub, "linker_id") or sub.linker_id is None:
            sub.linker_id = torch.zeros(sub.num_nodes, dtype=torch.long)
        sub.linker_metas: Dict[int, dict] = {}

        for lid, clone_source_in_M, anchor_in_M, is_u_side in per_frag_clones[frag_idx]:
            # Anchor atom is in this fragment.
            anchor_new = int(old_to_new[anchor_in_M].item())
            if anchor_new < 0:
                raise RuntimeError(
                    f"anchor atom {anchor_in_M} for cut {lid} not in fragment "
                    f"{frag_idx} after extraction — partition is inconsistent")

            # Append a clone of clone_source_in_M (the OTHER side's endpoint).
            clone_idx = _append_atom(sub, mol_data, clone_source_in_M)
            # Bond the clone to the anchor with the original cut bond features.
            _append_bond(sub, clone_idx, anchor_new,
                         cut_meta[lid]["cut_bond_features"])
            # Mask the clone and its incident edges.
            mask_linker_atom(sub, clone_idx)
            sub.linker_id[clone_idx] = lid

            if store_orig:
                # The clone is meant to represent the OTHER side's atom — its
                # original features (pre-mask) plus the cut bond features.
                src_features = cut_meta[lid]["v_atom_features"] if is_u_side \
                               else cut_meta[lid]["u_atom_features"]
                sub.linker_metas[lid] = {
                    "atom_features":     src_features,
                    "cut_bond_features": cut_meta[lid]["cut_bond_features"],
                    "anchor_neighbour":  anchor_new,
                }
            else:
                sub.linker_metas[lid] = {"anchor_neighbour": anchor_new}

        _ensure_molpla_attrs(sub)
        out.append(sub)

    return out


# ---------------------------------------------------------------------------
# attach
# ---------------------------------------------------------------------------

def attach_fragments(fragments: Sequence[Data],
                     restore_features: bool = True,
                     bond_features: Optional[Dict[int, Dict[str, int]]] = None,
                     ) -> Data:
    """Reassemble a list of masked-clone fragments into one molecule.

    Pairs each ``linker_id`` across the fragment list (every id must
    appear in **exactly two** fragments) and merges the corresponding
    masked clones so the original cut bond is restored.

    Algorithm sketch (for each ``linker_id = k``):
      1. Identify the two fragments holding a clone with id ``k``;
         call the lower-index one ``F_lo`` and the higher ``F_hi``.
      2. Identify the masked-clone atom positions ``c_lo`` and ``c_hi``
         and their unique anchor neighbours ``a_lo``, ``a_hi``
         (recorded in ``linker_metas[k]["anchor_neighbour"]``).
      3. Concatenate node tables. Redirect ``c_lo`` → ``a_hi`` so that
         the edge ``a_lo — c_lo`` becomes ``a_lo — a_hi`` — the restored
         cut bond. Drop ``c_lo`` and ``c_hi`` from the node list (plus
         all edges incident on either, after the redirect).
      4. Apply the stashed ``cut_bond_features`` to the restored edge.

    No atom features need restoring: the anchored model's "owner-side
    atom was masked, must be unmasked on attach" doesn't apply here —
    in the fragments paradigm every original atom stays unmasked, and
    only the fresh clones carry masked features.

    Parameters
    ----------
    fragments : sequence of Data
        The k fragments output by :func:`detach_fragments`.
    restore_features : bool, default True
        If True, restore each cut bond's original edge features from
        ``linker_metas``. If False, the restored bonds get default
        single-bond features.
    bond_features : dict[int, dict[str, int]], optional
        Per-``linker_id`` override for the restored bond's features.
        Wins over ``linker_metas`` when both are present.

    Returns
    -------
    Data
        The merged molecular graph (``num_nodes == sum(num_nodes) -
        2 * num_cuts`` — two clones drop per cut).
    """
    if len(fragments) == 0:
        raise ValueError("attach_fragments needs at least one fragment")
    if len(fragments) == 1:
        return fragments[0].clone()

    # ---- 1. discover all linker_ids in play, pair them across fragments --
    # id_to_pair[id] = list of (frag_idx, atom_local_idx)
    id_to_pair: Dict[int, List[Tuple[int, int]]] = {}
    for fi, frag in enumerate(fragments):
        if not hasattr(frag, "linker_id") or frag.linker_id is None:
            continue
        for local, lid in enumerate(frag.linker_id.tolist()):
            if lid <= 0:
                continue
            id_to_pair.setdefault(int(lid), []).append((fi, local))
    for lid, sites in id_to_pair.items():
        if len(sites) != 2:
            raise ValueError(
                f"linker_id={lid} must appear in exactly two fragments; "
                f"found {len(sites)} occurrences {sites}")

    # ---- 2. compute global node offset for each fragment -----------------
    offsets: List[int] = [0]
    for f in fragments[:-1]:
        offsets.append(offsets[-1] + f.num_nodes)
    total_atoms = offsets[-1] + fragments[-1].num_nodes

    # ---- 3. build concat tables ------------------------------------------
    out = MolPalleteData(num_nodes=total_atoms)
    for a in NODE_ATTRS:
        out[a] = torch.cat([f[a] for f in fragments])
    out.is_linker = torch.cat([f.is_linker for f in fragments])
    out.linker_id = torch.cat([
        f.linker_id if hasattr(f, "linker_id") and f.linker_id is not None
        else torch.zeros(f.num_nodes, dtype=torch.long)
        for f in fragments
    ])
    out.edge_index = torch.cat(
        [f.edge_index + off for f, off in zip(fragments, offsets)], dim=1)
    for a in EDGE_ATTRS:
        out[a] = torch.cat([f[a] for f in fragments])
    out.edge_is_linker = torch.cat([f.edge_is_linker for f in fragments])

    # ---- 4. resolve per-cut redirect + drop --------------------------------
    # Per linker_id: pick lower-index fragment as source ("redirect from"),
    # higher as sink ("its anchor is the merge target").
    redirect_map: Dict[int, int] = {}            # global src_clone -> sink_anchor_global
    drop_atoms: List[int] = []                   # global indices to drop entirely
    restored_pairs: List[Tuple[int, int, int]] = []  # (lid, a_lo_global, a_hi_global)

    for lid, sites in id_to_pair.items():
        sites.sort()  # by (frag_idx, atom_local_idx) — lower frag_idx first
        (lo_fi, lo_local), (hi_fi, hi_local) = sites
        lo_global = offsets[lo_fi] + lo_local
        hi_global = offsets[hi_fi] + hi_local

        lo_meta = (fragments[lo_fi].linker_metas or {}).get(lid, {})
        hi_meta = (fragments[hi_fi].linker_metas or {}).get(lid, {})
        if "anchor_neighbour" not in lo_meta or "anchor_neighbour" not in hi_meta:
            raise ValueError(
                f"fragment linker_metas[{lid}] missing 'anchor_neighbour' on "
                f"one or both sides; fragments must be produced by "
                f"detach_fragments to be reassemblable")
        a_lo_global = offsets[lo_fi] + int(lo_meta["anchor_neighbour"])
        a_hi_global = offsets[hi_fi] + int(hi_meta["anchor_neighbour"])

        # Redirect: lo_global -> a_hi_global. Edges a_lo--lo_global now read
        # as a_lo--a_hi (the restored cut bond).
        redirect_map[lo_global] = a_hi_global
        # Drop hi_global entirely (its edge to a_hi gets dropped with it).
        drop_atoms.append(lo_global)
        drop_atoms.append(hi_global)
        restored_pairs.append((lid, a_lo_global, a_hi_global))

    # ---- 5. apply redirect to edges ----------------------------------------
    src = out.edge_index[0].clone()
    dst = out.edge_index[1].clone()
    for orig, new in redirect_map.items():
        src[src == orig] = new
        dst[dst == orig] = new

    # ---- 6. drop edges that touch any to-be-dropped sink clone -----------
    # (Sources have already been redirected; their edges live on under their
    # new endpoints. Sinks were never redirected — drop their incident edges.)
    sink_clone_globals = set()
    for lid, sites in id_to_pair.items():
        _, hi = sorted(sites)
        sink_clone_globals.add(offsets[hi[0]] + hi[1])
    edge_keep_mask = ~(torch.isin(src, torch.tensor(sorted(sink_clone_globals),
                                                    dtype=torch.long))
                       | torch.isin(dst, torch.tensor(sorted(sink_clone_globals),
                                                      dtype=torch.long)))
    src = src[edge_keep_mask]
    dst = dst[edge_keep_mask]
    for a in EDGE_ATTRS:
        out[a] = out[a][edge_keep_mask]
    out.edge_is_linker = out.edge_is_linker[edge_keep_mask]

    # ---- 7. apply cut bond feature restoration ----------------------------
    bond_features = bond_features or {}
    for lid, a_lo_global, a_hi_global in restored_pairs:
        chosen = bond_features.get(lid)
        if chosen is None and restore_features:
            lo_meta = (fragments[id_to_pair[lid][0][0]].linker_metas or {}).get(lid, {})
            chosen = lo_meta.get("cut_bond_features")
        if chosen is None:
            chosen = _default_bond_features()
        # Locate the redirected edge (a_lo_global, a_hi_global) — typically two
        # directed entries in src/dst.
        matches = ((src == a_lo_global) & (dst == a_hi_global)) | \
                  ((src == a_hi_global) & (dst == a_lo_global))
        for e_idx in matches.nonzero(as_tuple=False).flatten().tolist():
            for a, val in chosen.items():
                out[a][e_idx] = val
            out.edge_is_linker[e_idx] = False

    # ---- 8. compact atom indices: drop the clone positions ---------------
    keep_mask = torch.ones(total_atoms, dtype=torch.bool)
    for a in drop_atoms:
        keep_mask[a] = False
    # Build old->new map.
    drop_counts = torch.zeros(total_atoms + 1, dtype=torch.long)
    for a in drop_atoms:
        drop_counts[a + 1] += 1
    drop_cum = torch.cumsum(drop_counts, dim=0)[:total_atoms]
    old_to_new = torch.arange(total_atoms) - drop_cum
    out.edge_index = torch.stack([old_to_new[src], old_to_new[dst]])

    for a in NODE_ATTRS:
        out[a] = out[a][keep_mask]
    out.is_linker = out.is_linker[keep_mask]
    out.linker_id = out.linker_id[keep_mask]
    out.num_nodes = int(keep_mask.sum().item())

    # ---- 9. finalise --------------------------------------------------------
    out.linker_metas = {}
    _ensure_molpla_attrs(out)
    return out


# ---------------------------------------------------------------------------
# helpers to construct a partition from a list of bonds
# ---------------------------------------------------------------------------

def partition_from_bonds(mol_or_data,
                         cut_bonds: Sequence[Tuple[int, int]],
                         ) -> FragmentPartition:
    """Build a :class:`FragmentPartition` from a sequence of bond-to-cut.

    Each entry of ``cut_bonds`` is ``(atom_a, atom_b)``; the function:

    1. Verifies each bond exists in M (and is not a ring bond — cutting a
       ring bond doesn't disconnect anything by itself).
    2. Computes the connected components of ``M`` with those bonds
       removed.
    3. Assigns each component a fragment index (sorted by smallest atom
       index for determinism) and emits one ``CutBond`` per input bond
       with sequential ``linker_id`` starting at 1.

    ``mol_or_data`` may be either an RDKit ``Mol`` or a PyG ``Data``
    (anything with ``edge_index`` works for the latter).

    Raises
    ------
    ValueError
        If any bond is missing or in a ring.
    """
    if hasattr(mol_or_data, "GetBonds"):
        return _partition_from_bonds_mol(mol_or_data, cut_bonds)
    else:
        return _partition_from_bonds_data(mol_or_data, cut_bonds)


def _partition_from_bonds_mol(mol, cut_bonds):
    n = mol.GetNumAtoms()
    cut_set = set()
    for a, b in cut_bonds:
        bond = mol.GetBondBetweenAtoms(int(a), int(b))
        if bond is None:
            raise ValueError(f"no bond between atoms {a} and {b} in mol")
        if bond.IsInRing():
            raise ValueError(
                f"bond {a}-{b} is in a ring; cutting it alone won't disconnect")
        cut_set.add(tuple(sorted((int(a), int(b)))))

    # Adjacency excluding cut bonds.
    adj: List[List[int]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if tuple(sorted((i, j))) in cut_set:
            continue
        adj[i].append(j); adj[j].append(i)

    return _build_partition_from_adj(adj, n, cut_set)


def _partition_from_bonds_data(data, cut_bonds):
    n = data.num_nodes
    cut_set = {tuple(sorted((int(a), int(b)))) for a, b in cut_bonds}
    adj: List[List[int]] = [[] for _ in range(n)]
    src, dst = data.edge_index[0].tolist(), data.edge_index[1].tolist()
    seen = set()
    for u, v in zip(src, dst):
        key = tuple(sorted((int(u), int(v))))
        if key in seen:
            continue
        seen.add(key)
        if key in cut_set:
            continue
        adj[key[0]].append(key[1]); adj[key[1]].append(key[0])
    return _build_partition_from_adj(adj, n, cut_set)


def _build_partition_from_adj(adj, n, cut_set):
    # Connected components.
    visited = [False] * n
    components: List[List[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        comp: List[int] = []
        stack = [start]
        while stack:
            a = stack.pop()
            if visited[a]:
                continue
            visited[a] = True
            comp.append(a)
            for nb in adj[a]:
                if not visited[nb]:
                    stack.append(nb)
        components.append(sorted(comp))

    # Order components by smallest atom index for determinism.
    components.sort(key=lambda c: c[0])
    atom_to_frag = [0] * n
    fragments = []
    for fi, comp in enumerate(components):
        for a in comp:
            atom_to_frag[a] = fi
        fragments.append(FragmentInfo(atoms_in_M=tuple(comp)))

    cuts = []
    for k, (u, v) in enumerate(sorted(cut_set), start=1):
        cuts.append(CutBond(
            linker_id=k,
            u_atom_in_M=u,
            v_atom_in_M=v,
            u_frag=atom_to_frag[u],
            v_frag=atom_to_frag[v],
        ))

    return FragmentPartition(
        fragments=tuple(fragments),
        cut_bonds=tuple(cuts),
    )


__all__ = [
    "detach_fragments", "attach_fragments",
    "partition_from_bonds", "subgraph_hash",
]
