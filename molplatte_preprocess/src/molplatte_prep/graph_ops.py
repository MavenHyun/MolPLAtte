"""Elementary detach / attach operations on masked-linker PyG molecular graphs.

Semantics of the *masked linker joint* (aligned with MolPLA &
``molpla_old/step3_pyg_data_conversion.py``):

Per cut bond ``A -- B`` (A on the core side, B on the R-group side):

* The **linker atom** is ``A``: a single heavy atom of M sitting on the core
  side of the cut.
* Detach "punches" the shared linker atom: ``A`` stays in the template (with
  its node attributes and every incident edge's bond attributes overwritten
  by reserved MASK indices); a **clone** ``A'`` of ``A`` is appended to the
  R-group, the original ``A--B`` bond is re-routed to ``A'--B``, and then
  ``A'`` is also masked.
* ``B`` and the rest of the R-group are **unmodified** — their chemistry is
  preserved.
* Attach reverses the operation by **merging** A and A': the R-group's clone
  is redirected to the template's A, and the now-restored cut bond
  ``A--B`` appears in the merged graph. The merged graph has
  ``n_template + n_rgroup - 1`` atoms.

Multiple joints per template (``detach_rgroups_multi``):

  A template with k masked joints carries a ``linker_id`` LongTensor (0 for
  non-linker atoms, positive integers identifying each joint). Each detached
  R-group carries a matching ``linker_id`` on its clone. ``attach_rgroup``
  accepts ``template_linker_id`` / ``rgroup_linker_id`` to disambiguate when
  multiple joints exist; with both inputs carrying ids it auto-matches by id.

``store_orig=True`` (default) snapshots the pre-mask atom / edge features
into ``data.linker_metas`` (dict keyed by linker_id) so that
``attach_rgroup`` can do a lossless round-trip. ``data.linker_meta`` is also
exposed for single-joint inputs as a backward-compat alias to
``linker_metas[1]``.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from rdkit import Chem
from torch_geometric.data import Data

from .graph_hash import subgraph_hash  # noqa: F401  (re-exported)
from .mol_features import (
    EDGE_ATTRS,
    MASK_VALUES,
    MolPalleteData,
    NODE_ATTRS,
    RDKIT_FEATURES,
)


def _ensure_molpla_attrs(data: Data) -> Data:
    """Guarantee a stable attribute schema on every Data we emit.

    Sets ``linker_id`` (zeros), ``linker_atom`` (empty long tensor), and
    ``linker_metas`` (empty dict) when missing, so heterogeneous batches
    of clean mols vs. masked templates share the same attr layout.

    Also derives ``linker_atom`` from ``linker_id`` (sorted by id) so it is
    always consistent with the per-atom id tensor.
    """
    n = data.num_nodes
    if not hasattr(data, "linker_id") or data.linker_id is None:
        data.linker_id = torch.zeros(n, dtype=torch.long)
    if not hasattr(data, "linker_metas") or data.linker_metas is None:
        data.linker_metas = {}

    # Derive linker_atom: atom indices where linker_id > 0, sorted by id.
    nonzero = (data.linker_id > 0).nonzero(as_tuple=False).flatten()
    if nonzero.numel() > 0:
        ids_at = data.linker_id[nonzero]
        order = torch.argsort(ids_at)
        data.linker_atom = nonzero[order]
    else:
        data.linker_atom = torch.zeros(0, dtype=torch.long)
    return data


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def is_linker_atom(data: Data, idx: int) -> bool:
    return bool(data.is_linker[idx].item())


def _atom_feature_snapshot(data: Data, idx: int) -> Dict[str, int]:
    return {a: int(data[a][idx].item()) for a in NODE_ATTRS}


def _edge_feature_snapshot(data: Data, e_idx: int) -> Dict[str, int]:
    return {a: int(data[a][e_idx].item()) for a in EDGE_ATTRS}


def _find_edge_indices(edge_index: torch.Tensor, u: int, v: int) -> List[int]:
    """Return positions of all (u,v) directed edges (typically two)."""
    mask = ((edge_index[0] == u) & (edge_index[1] == v)) | \
           ((edge_index[0] == v) & (edge_index[1] == u))
    return mask.nonzero(as_tuple=False).flatten().tolist()


def mask_linker_atom(data: Data, atom_idx: int, in_place: bool = True) -> Data:
    """Mask one atom's features + every incident edge's bond features.

    Sets ``is_linker[atom_idx]=True`` and ``edge_is_linker=True`` on each
    incident edge. The node and edge attribute values are overwritten with
    the reserved MASK indices.
    """
    if not in_place:
        data = data.clone()
    for a in NODE_ATTRS:
        data[a][atom_idx] = MASK_VALUES[a]
    data.is_linker[atom_idx] = True

    incident = (data.edge_index[0] == atom_idx) | (data.edge_index[1] == atom_idx)
    for a in EDGE_ATTRS:
        data[a][incident] = MASK_VALUES[a]
    data.edge_is_linker[incident] = True
    return data


def _ensure_linker_id(data: Data) -> None:
    """Lazily create ``data.linker_id`` (LongTensor, num_nodes) if absent."""
    if not hasattr(data, "linker_id") or data.linker_id is None:
        data.linker_id = torch.zeros(data.num_nodes, dtype=torch.long)


def _ensure_linker_metas(data: Data) -> None:
    if not hasattr(data, "linker_metas") or data.linker_metas is None:
        data.linker_metas = {}


# ---------------------------------------------------------------------------
# sub-graph extraction
# ---------------------------------------------------------------------------

def _extract_subgraph(data: Data, keep_atoms: Sequence[int]
                      ) -> Tuple[Data, "torch.Tensor", "torch.Tensor"]:
    """Return a fresh ``Data`` containing only ``keep_atoms``.

    Edges with any endpoint outside ``keep_atoms`` are dropped. All
    node/edge attributes are copied, including ``linker_id`` if present.

    Returns
    -------
    out : Data
    old_to_new : torch.LongTensor of size ``data.num_nodes``
        -1 for atoms not kept, else the atom's new index in ``out``.
    surviving_edge_idx : torch.LongTensor
        Indices into ``data.edge_index`` for the surviving edges.
    """
    keep_atoms = list(keep_atoms)
    keep_tensor = torch.as_tensor(keep_atoms, dtype=torch.long)
    n_in = data.num_nodes
    n_out = keep_tensor.numel()

    # Vectorised boolean mask + index remap (no Python edge loop).
    keep_node_mask = torch.zeros(n_in, dtype=torch.bool)
    keep_node_mask[keep_tensor] = True
    old_to_new = torch.full((n_in,), -1, dtype=torch.long)
    old_to_new[keep_tensor] = torch.arange(n_out, dtype=torch.long)

    src, dst = data.edge_index[0], data.edge_index[1]
    edge_keep_mask = keep_node_mask[src] & keep_node_mask[dst]
    surviving_edge_idx = edge_keep_mask.nonzero(as_tuple=False).flatten()

    if surviving_edge_idx.numel() > 0:
        new_edge_index = torch.stack([
            old_to_new[src[surviving_edge_idx]],
            old_to_new[dst[surviving_edge_idx]],
        ])
    else:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)

    out = MolPalleteData(edge_index=new_edge_index, num_nodes=n_out)
    for a in NODE_ATTRS:
        out[a] = data[a][keep_tensor].clone()
    out.is_linker = data.is_linker[keep_tensor].clone()
    # Carry linker_id forward if the input has one.
    if hasattr(data, "linker_id") and data.linker_id is not None:
        out.linker_id = data.linker_id[keep_tensor].clone()
    for a in EDGE_ATTRS:
        out[a] = data[a][surviving_edge_idx].clone()
    out.edge_is_linker = data.edge_is_linker[surviving_edge_idx].clone()
    return out, old_to_new, surviving_edge_idx


def _append_atom(target: Data, source: Data, source_idx: int) -> int:
    """Append a clone of ``source[source_idx]`` to ``target``.

    Returns the new atom's index in ``target``. If ``target`` already carries
    a ``linker_id`` tensor, it is extended by one zero element so it stays
    aligned with ``num_nodes``.
    """
    new_idx = target.num_nodes
    for a in NODE_ATTRS:
        target[a] = torch.cat(
            [target[a], source[a][source_idx:source_idx + 1].clone()])
    target.is_linker = torch.cat(
        [target.is_linker, torch.tensor([False], dtype=torch.bool)])
    if hasattr(target, "linker_id") and target.linker_id is not None:
        target.linker_id = torch.cat(
            [target.linker_id, torch.zeros(1, dtype=torch.long)])
    target.num_nodes = new_idx + 1
    return new_idx


def _append_bond(target: Data, u: int, v: int,
                 features: Dict[str, int]) -> None:
    """Append a fresh bidirectional bond between ``u`` and ``v`` in ``target``."""
    new_src = torch.tensor([u, v], dtype=torch.long)
    new_dst = torch.tensor([v, u], dtype=torch.long)
    new_edges = torch.stack([new_src, new_dst])
    target.edge_index = torch.cat([target.edge_index, new_edges], dim=1)
    for a in EDGE_ATTRS:
        val = features[a]
        target[a] = torch.cat(
            [target[a], torch.tensor([val, val], dtype=target[a].dtype)])
    target.edge_is_linker = torch.cat(
        [target.edge_is_linker, torch.tensor([False, False], dtype=torch.bool)])


# ---------------------------------------------------------------------------
# snapshot originals (used by both single and multi detach)
# ---------------------------------------------------------------------------

def _snapshot_in_core_incident(mol_data: Data, core_linker: int,
                               core_set: set, exclude_atoms: set
                               ) -> List[dict]:
    """Bonds incident on ``core_linker`` that stay inside ``core_set``.

    ``exclude_atoms`` should contain the R-group-side endpoint(s) of cut
    bonds (so we skip those edges — they are not in the template).
    """
    out: List[dict] = []
    seen = set()
    for e in range(mol_data.edge_index.size(1)):
        u = int(mol_data.edge_index[0, e].item())
        v = int(mol_data.edge_index[1, e].item())
        if (u == core_linker) ^ (v == core_linker):
            other = v if u == core_linker else u
            if other in core_set and other not in exclude_atoms:
                key = tuple(sorted((u, v)))
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "neighbour": other,
                    "features":  _edge_feature_snapshot(mol_data, e),
                })
    return out


# ---------------------------------------------------------------------------
# detach (single R-group)
# ---------------------------------------------------------------------------

def detach_rgroup(mol_data: Data,
                  rgroup_atoms: Iterable[int],
                  core_linker: int,
                  rgroup_linker: int,
                  store_orig: bool = True,
                  linker_id: Optional[int] = None,
                  ) -> Tuple[Data, Data]:
    """Detach a single R-group along the cut bond
    ``core_linker``--``rgroup_linker``. Thin wrapper around
    ``detach_rgroups_multi`` so the two share the same semantics.

    ``linker_id`` (default ``None``) controls the id written at the joint
    position. When ``None``, the next available id is used: ``1`` if the
    input has no prior linker_id, else ``max(existing) + 1`` — so chaining
    detaches on a template never collides.
    """
    ids = [int(linker_id)] if linker_id is not None else None
    template, rgroups = detach_rgroups_multi(
        mol_data,
        rgroup_infos=[(tuple(int(x) for x in rgroup_atoms),
                       int(core_linker), int(rgroup_linker))],
        store_orig=store_orig,
        ids=ids,
    )
    return template, rgroups[0]


# ---------------------------------------------------------------------------
# detach (multi R-group, single pass)
# ---------------------------------------------------------------------------

def detach_rgroups_multi(mol_data: Data,
                         rgroup_infos: Sequence[Tuple[Iterable[int], int, int]],
                         store_orig: bool = True,
                         ids: Optional[Sequence[int]] = None,
                         ) -> Tuple[Data, List[Data]]:
    """Detach k R-groups from ``mol_data`` in a single pass.

    ``rgroup_infos`` is a sequence of ``(rgroup_atoms, core_linker,
    rgroup_linker)`` triples, all in M's original atom indexing. Each
    R-group's atom set must be disjoint from every other R-group.

    ``ids`` (optional) lets the caller pick the linker_id assigned to each
    joint. Defaults to ``[1, 2, ..., k]``.

    Returns
    -------
    template : Data
        Core sub-graph of M (i.e. M minus the union of all R-group atoms),
        with each ``core_linker[i]`` masked and ``template.linker_id``
        carrying ``ids[i]`` at that position.
    rgroups : list[Data]
        Each entry is one detached R-group. The R-group atoms stay
        unmasked, a clone of the corresponding ``core_linker`` is appended
        (masked), and ``rgroup.linker_id`` carries the same id as the
        template's matching joint.
    """
    n = mol_data.num_nodes
    k = len(rgroup_infos)
    if k == 0:
        raise ValueError("rgroup_infos must be non-empty")

    # ---- existing-joint state on the input (if any) ---------------------
    has_existing_id = (hasattr(mol_data, "linker_id")
                       and mol_data.linker_id is not None)
    existing_max_id = int(mol_data.linker_id.max().item()) if has_existing_id else 0
    existing_link_atoms: set = set()
    if has_existing_id:
        existing_link_atoms = set(
            (mol_data.linker_id > 0).nonzero(as_tuple=False).flatten().tolist()
        )

    # ---- validate + collect ----
    parsed = []
    all_rg_atoms: set = set()
    seen_core_linkers: set = set()
    for i, (rg_atoms, core_l, rg_l) in enumerate(rgroup_infos):
        rg_set = set(int(x) for x in rg_atoms)
        core_l = int(core_l)
        rg_l = int(rg_l)
        if rg_l not in rg_set:
            raise ValueError(f"rgroup_linker {rg_l} not in rgroup_atoms[{i}]")
        if core_l in rg_set:
            raise ValueError(f"core_linker {core_l} must not be in rgroup_atoms[{i}]")
        if rg_set & all_rg_atoms:
            raise ValueError(
                f"R-group {i} overlaps with a previous R-group at atoms "
                f"{rg_set & all_rg_atoms}")
        if existing_link_atoms & rg_set:
            raise ValueError(
                f"R-group {i} overlaps existing masked-joint atoms "
                f"{existing_link_atoms & rg_set}; cannot detach atoms that "
                f"are already in masked-linker state.")
        if core_l in existing_link_atoms:
            raise ValueError(
                f"core_linker {core_l} is already a masked joint of the input")
        if core_l in seen_core_linkers:
            raise ValueError(
                f"core_linker {core_l} is shared by multiple R-groups in this "
                f"call (geminal joints). The current data model represents at "
                f"most one joint per template atom; if you really need this, "
                f"chain single-cut detaches (each adds a clone to a different "
                f"R-group, not to the template).")
        seen_core_linkers.add(core_l)
        all_rg_atoms |= rg_set
        cut_e = _find_edge_indices(mol_data.edge_index, core_l, rg_l)
        if not cut_e:
            raise ValueError(
                f"No bond between core_linker={core_l} and rgroup_linker={rg_l}")
        parsed.append((rg_set, core_l, rg_l, cut_e[0]))

    core_atoms = sorted(set(range(n)) - all_rg_atoms)
    core_set = set(core_atoms)

    if ids is None:
        ids = list(range(existing_max_id + 1, existing_max_id + 1 + k))
    if len(ids) != k:
        raise ValueError("ids must have the same length as rgroup_infos")
    if any(x <= 0 for x in ids) or len(set(ids)) != k:
        raise ValueError("ids must be positive integers and unique")
    if has_existing_id:
        clash = set(ids) & set(mol_data.linker_id.tolist())
        clash.discard(0)
        if clash:
            raise ValueError(
                f"ids {clash} collide with existing linker_ids in the input")

    # ---- snapshot originals BEFORE any masking ----
    snap_atom_orig: List[Dict[str, int]] = []
    snap_cut_bond: List[Dict[str, int]] = []
    snap_in_core_incident: List[List[dict]] = []
    exclude_for_incident = {rg_l for (_s, _c, rg_l, _e) in parsed}
    for (_rg_set, core_l, _rg_l, cut_e_idx) in parsed:
        snap_atom_orig.append(_atom_feature_snapshot(mol_data, core_l))
        snap_cut_bond.append(_edge_feature_snapshot(mol_data, cut_e_idx))
        snap_in_core_incident.append(
            _snapshot_in_core_incident(mol_data, core_l, core_set,
                                       exclude_atoms=exclude_for_incident))

    # ---------------- TEMPLATE -------------------------------------------
    template, core_old_to_new, _ = _extract_subgraph(mol_data, core_atoms)
    # Carry forward (and reindex) existing linker_id / linker_metas.
    if has_existing_id:
        existing_metas = getattr(mol_data, "linker_metas", {}) or {}
        # _extract_subgraph already copied linker_id; now remap incident-feature
        # neighbour indices for each existing meta.
        preserved_metas: Dict[int, dict] = {}
        for lid, meta in existing_metas.items():
            new_incident = []
            for rec in meta.get("incident_features", []):
                nbr_new = int(core_old_to_new[rec["neighbour"]].item())
                if nbr_new < 0:
                    continue  # neighbour atom no longer in the template
                new_incident.append({"neighbour": nbr_new,
                                     "features": rec["features"]})
            preserved_metas[lid] = {
                "atom_features":     meta["atom_features"],
                "incident_features": new_incident,
                "cut_bond_features": meta["cut_bond_features"],
            }
        template.linker_metas = preserved_metas
    else:
        template.linker_id = torch.zeros(template.num_nodes, dtype=torch.long)
        template.linker_metas: Dict[int, dict] = {}

    for (_rg_set, core_l, _rg_l, _), lid, atom_orig, cut_orig, incident_orig in zip(
            parsed, ids, snap_atom_orig, snap_cut_bond, snap_in_core_incident):
        new_core_l = int(core_old_to_new[core_l].item())
        mask_linker_atom(template, new_core_l)
        template.linker_id[new_core_l] = lid
        if store_orig:
            remapped_incident = []
            for rec in incident_orig:
                nbr_new = int(core_old_to_new[rec["neighbour"]].item())
                if nbr_new >= 0:
                    remapped_incident.append({"neighbour": nbr_new,
                                              "features":  rec["features"]})
            template.linker_metas[lid] = {
                "atom_features":     atom_orig,
                "incident_features": remapped_incident,
                "cut_bond_features": cut_orig,
            }

    _ensure_molpla_attrs(template)

    # ---------------- R-GROUPS ------------------------------------------
    rgroups_out: List[Data] = []
    for (rg_set, core_l, rg_l, _), lid, atom_orig, cut_orig in zip(
            parsed, ids, snap_atom_orig, snap_cut_bond):
        rg_atoms_sorted = sorted(rg_set)
        rgroup, rg_old_to_new, _ = _extract_subgraph(mol_data, rg_atoms_sorted)
        new_rg_neighbour = int(rg_old_to_new[rg_l].item())

        clone_idx = _append_atom(rgroup, mol_data, core_l)
        _append_bond(rgroup, clone_idx, new_rg_neighbour, cut_orig)
        mask_linker_atom(rgroup, clone_idx)

        # If the sub-graph carried over an existing linker_id from the input,
        # extend it; otherwise create a fresh zero tensor.
        if not hasattr(rgroup, "linker_id") or rgroup.linker_id is None:
            rgroup.linker_id = torch.zeros(rgroup.num_nodes, dtype=torch.long)
        elif rgroup.linker_id.numel() != rgroup.num_nodes:
            rgroup.linker_id = torch.cat([rgroup.linker_id,
                                          torch.zeros(1, dtype=torch.long)])
        rgroup.linker_id[clone_idx] = lid

        if store_orig:
            rgroup.linker_metas = {lid: {
                "atom_features":     atom_orig,
                "incident_features": [],
                "cut_bond_features": cut_orig,
                "rgroup_neighbour":  new_rg_neighbour,
            }}
        else:
            rgroup.linker_metas = {}

        _ensure_molpla_attrs(rgroup)
        rgroups_out.append(rgroup)

    return template, rgroups_out


# ---------------------------------------------------------------------------
# attach
# ---------------------------------------------------------------------------

def _default_bond_features() -> Dict[str, int]:
    return {
        "bond_type":        RDKIT_FEATURES["bond_type"].index(Chem.rdchem.BondType.SINGLE),
        "edge_is_aromatic": RDKIT_FEATURES["edge_is_aromatic"].index(False),
        "is_conjugated":    RDKIT_FEATURES["is_conjugated"].index(False),
        "bond_dir":         RDKIT_FEATURES["bond_dir"].index(Chem.rdchem.BondDir.NONE),
        "bond_stereo":      RDKIT_FEATURES["bond_stereo"].index(Chem.rdchem.BondStereo.STEREONONE),
    }


def _resolve_linker(data: Data,
                    idx_hint: Optional[int] = None,
                    id_hint:  Optional[int] = None) -> Tuple[int, Optional[int]]:
    """Return ``(atom_index, linker_id_or_None)`` for the desired joint."""
    has_id = hasattr(data, "linker_id") and data.linker_id is not None

    if idx_hint is not None and id_hint is not None:
        raise ValueError("Provide only one of *_linker_idx and *_linker_id")

    if id_hint is not None:
        if not has_id:
            raise ValueError("Graph has no linker_id tensor")
        matches = (data.linker_id == int(id_hint)).nonzero(as_tuple=False).flatten()
        if matches.numel() == 0:
            raise ValueError(f"No linker atom with id={id_hint}")
        if matches.numel() > 1:
            raise ValueError(f"Multiple linker atoms share id={id_hint}: {matches.tolist()}")
        idx = int(matches.item())
        if not is_linker_atom(data, idx):
            raise ValueError(f"Atom {idx} has linker_id={id_hint} but is_linker=False")
        return idx, int(id_hint)

    if idx_hint is not None:
        idx = int(idx_hint)
        if not is_linker_atom(data, idx):
            raise ValueError(f"Atom {idx} is not flagged is_linker=True")
        lid = int(data.linker_id[idx].item()) if has_id else None
        return idx, lid

    # Auto-detect: only valid if there is exactly one is_linker atom.
    flags = data.is_linker.nonzero(as_tuple=False).flatten()
    if flags.numel() == 1:
        idx = int(flags.item())
        lid = int(data.linker_id[idx].item()) if has_id else None
        return idx, lid
    if flags.numel() == 0:
        raise ValueError("No is_linker atom in graph; pass an explicit index or id")
    raise ValueError(
        f"Multiple linker atoms in graph ({flags.tolist()}); "
        f"disambiguate via the *_linker_id or *_linker_idx argument")


def _meta_for_id(data: Data, lid: Optional[int]) -> Optional[dict]:
    """Look up ``data.linker_metas[lid]`` if present, else ``None``."""
    if lid is None:
        return None
    metas = getattr(data, "linker_metas", None)
    if metas is None:
        return None
    return metas.get(lid)


def attach_rgroup(template_data: Data,
                  rgroup_data: Data,
                  template_linker_idx: Optional[int] = None,
                  rgroup_linker_idx:   Optional[int] = None,
                  template_linker_id:  Optional[int] = None,
                  rgroup_linker_id:    Optional[int] = None,
                  restore_features: bool = True,
                  bond_features: Optional[Dict[str, int]] = None,
                  ) -> Data:
    """Re-attach a masked R-group onto a masked template by merging the
    shared linker atom.

    Multi-joint disambiguation precedence:

    1. ``template_linker_id`` / ``rgroup_linker_id`` (recommended for
       multi-joint templates).
    2. ``template_linker_idx`` / ``rgroup_linker_idx`` (literal atom
       indices).
    3. **Auto-match by id.** If neither hint is given but both inputs carry
       ``linker_id`` and exactly one id is shared, that id is used.
    4. **Auto-detect.** Falls back to the unique ``is_linker=True`` atom on
       each side; raises if either side is ambiguous.

    Other parameters work as before:
    - ``restore_features`` (default True): restore the joint's atom and
      incident-bond features from ``linker_metas``.
    - ``bond_features``: explicit override for the cut bond's attributes.
    """
    # ---- step 1: pick the matching joint on each side --------------------
    t_link_id_for_match = template_linker_id
    r_link_id_for_match = rgroup_linker_id
    if t_link_id_for_match is None and r_link_id_for_match is None \
       and template_linker_idx is None and rgroup_linker_idx is None:
        # auto-match by id if both have linker_id tensors and share an id
        t_has = hasattr(template_data, "linker_id") and template_data.linker_id is not None
        r_has = hasattr(rgroup_data,   "linker_id") and rgroup_data.linker_id   is not None
        if t_has and r_has:
            t_ids = set(template_data.linker_id[template_data.linker_id > 0].tolist())
            r_ids = set(rgroup_data.linker_id[rgroup_data.linker_id > 0].tolist())
            shared = t_ids & r_ids
            if len(shared) == 1:
                t_link_id_for_match = r_link_id_for_match = next(iter(shared))

    t_link, t_lid = _resolve_linker(template_data,
                                    idx_hint=template_linker_idx,
                                    id_hint=t_link_id_for_match)
    r_link_local, r_lid = _resolve_linker(rgroup_data,
                                          idx_hint=rgroup_linker_idx,
                                          id_hint=r_link_id_for_match)

    n_t = template_data.num_nodes
    n_r = rgroup_data.num_nodes
    r_link = r_link_local + n_t

    # ---- step 2: concat nodes -------------------------------------------
    out = MolPalleteData(num_nodes=n_t + n_r)
    for a in NODE_ATTRS:
        out[a] = torch.cat([template_data[a], rgroup_data[a]])
    out.is_linker = torch.cat([template_data.is_linker, rgroup_data.is_linker])

    # carry linker_id forward; ids in the rgroup half get propagated, but
    # the merged joint will be cleared at the end.
    t_lid_tensor = (template_data.linker_id
                    if hasattr(template_data, "linker_id") and template_data.linker_id is not None
                    else torch.zeros(n_t, dtype=torch.long))
    r_lid_tensor = (rgroup_data.linker_id
                    if hasattr(rgroup_data, "linker_id") and rgroup_data.linker_id is not None
                    else torch.zeros(n_r, dtype=torch.long))
    out.linker_id = torch.cat([t_lid_tensor, r_lid_tensor])

    # ---- step 3: concat edges (R-group offset) --------------------------
    rg_edge_index = rgroup_data.edge_index + n_t
    edge_index = torch.cat([template_data.edge_index, rg_edge_index], dim=1)
    edge_attrs: Dict[str, torch.Tensor] = {}
    for a in EDGE_ATTRS:
        edge_attrs[a] = torch.cat([template_data[a], rgroup_data[a]])
    edge_attrs["edge_is_linker"] = torch.cat(
        [template_data.edge_is_linker, rgroup_data.edge_is_linker])
    out.edge_index = edge_index
    for k, v in edge_attrs.items():
        out[k] = v

    # ---- step 4: restore features at the joint (still pre-merge) --------
    t_meta = _meta_for_id(template_data, t_lid)
    r_meta = _meta_for_id(rgroup_data,   r_lid)

    if restore_features:
        if t_meta is not None:
            for a, val in t_meta["atom_features"].items():
                out[a][t_link] = val
            out.is_linker[t_link] = False
            for rec in t_meta["incident_features"]:
                nbr = rec["neighbour"]
                for e_idx in _find_edge_indices(out.edge_index, t_link, nbr):
                    for a, val in rec["features"].items():
                        out[a][e_idx] = val
                    out.edge_is_linker[e_idx] = False

        chosen_bond = bond_features
        if chosen_bond is None:
            chosen_bond = (r_meta or t_meta or {}).get("cut_bond_features") \
                          or _default_bond_features()
        for e_idx in range(out.edge_index.size(1)):
            u = int(out.edge_index[0, e_idx].item())
            v = int(out.edge_index[1, e_idx].item())
            if u == r_link or v == r_link:
                for a, val in chosen_bond.items():
                    out[a][e_idx] = val
                out.edge_is_linker[e_idx] = False

        out.is_linker[r_link] = False
    else:
        chosen_bond = bond_features or _default_bond_features()
        for e_idx in range(out.edge_index.size(1)):
            u = int(out.edge_index[0, e_idx].item())
            v = int(out.edge_index[1, e_idx].item())
            if u == r_link or v == r_link:
                for a, val in chosen_bond.items():
                    out[a][e_idx] = val
                out.edge_is_linker[e_idx] = False

    # ---- step 5: merge A and A' (redirect r_link to t_link, drop r_link) -
    out.edge_index[0, out.edge_index[0] == r_link] = t_link
    out.edge_index[1, out.edge_index[1] == r_link] = t_link

    keep_mask = torch.ones(n_t + n_r, dtype=torch.bool)
    keep_mask[r_link] = False
    old_to_new = torch.arange(n_t + n_r) - (torch.arange(n_t + n_r) > r_link).long()
    out.edge_index[0] = old_to_new[out.edge_index[0]]
    out.edge_index[1] = old_to_new[out.edge_index[1]]

    for a in NODE_ATTRS:
        out[a] = out[a][keep_mask]
    out.is_linker = out.is_linker[keep_mask]
    out.linker_id = out.linker_id[keep_mask]
    # The merged template atom keeps t_link's position but no longer is a
    # linker; clear its id too.
    out.linker_id[t_link] = 0
    out.num_nodes = (n_t + n_r) - 1

    # Propagate linker_metas for any remaining joints in the template.
    out.linker_metas = {}
    if hasattr(template_data, "linker_metas") and template_data.linker_metas is not None:
        out.linker_metas = {lid: meta
                            for lid, meta in template_data.linker_metas.items()
                            if lid != t_lid}

    _ensure_molpla_attrs(out)
    return out


# ---------------------------------------------------------------------------
# attach (k R-groups at the same time — chemically safe alternative
#                                      to iterating attach_rgroup)
# ---------------------------------------------------------------------------

def attach_rgroups(template_data: Data,
                   rgroups: Sequence[Data],
                   linker_ids: Optional[Sequence[int]] = None,
                   restore_features: bool = True,
                   bond_features: Optional[Dict[int, Dict[str, int]]] = None,
                   ) -> Data:
    """Re-attach **k R-groups simultaneously** to a multi-joint template.

    Iterating :func:`attach_rgroup` k times works for the case where every
    joint sits on a distinct atom, but it creates intermediate Data objects
    whose ``*``-decorated state is a chemical fragment rather than a
    complete molecule. This function does all merges in a single pass, so:

    1. The intermediate ``*``-fragment state never exists — masked-joint
       atoms are restored only when all clones have been redirected to
       them.
    2. Two joints that share the same template atom (e.g. geminal
       disubstituents on one carbon) are handled correctly, because we
       redirect both clones to the shared atom in one shot.

    Parameters
    ----------
    template_data : Data
        Template with k masked joints (each carrying a ``linker_id`` > 0
        and an entry in ``linker_metas``).
    rgroups : sequence of Data
        Exactly one R-group per joint. Each must carry a single
        ``is_linker=True`` clone atom whose ``linker_id`` matches one of
        the template's joints. Order need not match the joint ids — we
        resolve by id.
    linker_ids : sequence of int, optional
        Explicit pairing of ``rgroups[i]`` with the template joint of that
        id. Defaults to auto-match by each R-group's own ``linker_id``.
    restore_features, bond_features
        Same semantics as :func:`attach_rgroup`. ``bond_features`` is now a
        ``dict[linker_id -> dict[str, int]]`` to allow per-joint overrides.
    """
    if len(rgroups) == 0:
        return template_data.clone()

    n_t = template_data.num_nodes
    n_rs = [rg.num_nodes for rg in rgroups]

    # ---- pair each R-group with a template joint by linker_id ---------
    if linker_ids is None:
        linker_ids = []
        for rg in rgroups:
            ids = rg.linker_id[rg.linker_id > 0]
            if ids.numel() != 1:
                raise ValueError(
                    f"R-group must have exactly one masked clone; got "
                    f"linker_id values {rg.linker_id.tolist()}")
            linker_ids.append(int(ids.item()))
    if len(linker_ids) != len(rgroups):
        raise ValueError("linker_ids must have the same length as rgroups")

    template_joint_ids = set(template_data.linker_id[template_data.linker_id > 0].tolist())
    for lid in linker_ids:
        if lid not in template_joint_ids:
            raise ValueError(
                f"linker_id={lid} not present in template "
                f"(joints {sorted(template_joint_ids)})")
    if len(set(linker_ids)) != len(linker_ids):
        raise ValueError("linker_ids must be unique across rgroups")

    # ---- concat nodes ----
    out = MolPalleteData(num_nodes=n_t + sum(n_rs))
    for a in NODE_ATTRS:
        out[a] = torch.cat([template_data[a]] + [rg[a] for rg in rgroups])
    out.is_linker = torch.cat([template_data.is_linker] + [rg.is_linker for rg in rgroups])
    out.linker_id = torch.cat(
        [template_data.linker_id] + [rg.linker_id for rg in rgroups])

    # ---- concat edges (offset each R-group's edge_index) ----
    offsets = [n_t]
    for n in n_rs[:-1]:
        offsets.append(offsets[-1] + n)
    edge_indices = [template_data.edge_index] + [
        rg.edge_index + off for rg, off in zip(rgroups, offsets)
    ]
    out.edge_index = torch.cat(edge_indices, dim=1)
    for a in EDGE_ATTRS:
        out[a] = torch.cat([template_data[a]] + [rg[a] for rg in rgroups])
    out.edge_is_linker = torch.cat(
        [template_data.edge_is_linker] + [rg.edge_is_linker for rg in rgroups])

    # ---- map each linker_id -> (t_link in template, r_link in concat) ----
    pairs: List[Tuple[int, int, int]] = []   # (lid, t_link_global, r_link_global)
    for lid, rg, off in zip(linker_ids, rgroups, offsets):
        t_mask = (template_data.linker_id == lid).nonzero(as_tuple=False).flatten()
        if t_mask.numel() == 0:
            raise ValueError(f"no template atom carries linker_id={lid}")
        # If multiple atoms share lid (shouldn't happen but guard), take first.
        t_link_global = int(t_mask[0].item())
        r_local = (rg.linker_id > 0).nonzero(as_tuple=False).flatten()
        r_link_global = int(r_local.item()) + off
        pairs.append((lid, t_link_global, r_link_global))

    # ---- step A: restore atom features for every joint --------------
    if restore_features:
        for lid, t_link, _ in pairs:
            meta = (template_data.linker_metas or {}).get(lid)
            if meta is None:
                continue
            for a, val in meta["atom_features"].items():
                out[a][t_link] = val
            out.is_linker[t_link] = False
            for rec in meta.get("incident_features", []):
                nbr = rec["neighbour"]
                for e_idx in _find_edge_indices(out.edge_index, t_link, nbr):
                    for a, val in rec["features"].items():
                        out[a][e_idx] = val
                    out.edge_is_linker[e_idx] = False

    # ---- step B: restore each cut bond on the corresponding A'-B edge ----
    # bond_features overrides per joint take precedence; else the rgroup's
    # cut_bond_features; else the default single bond.
    bond_features = bond_features or {}
    for lid, _, r_link in pairs:
        # Find which rgroup carried this joint
        rg_idx = linker_ids.index(lid)
        rg_meta = (rgroups[rg_idx].linker_metas or {}).get(lid)
        chosen = bond_features.get(lid)
        if chosen is None and rg_meta is not None:
            chosen = rg_meta.get("cut_bond_features")
        if chosen is None:
            chosen = _default_bond_features()
        # Edges incident to r_link in the merged graph (in the as-concatenated
        # indexing) get the cut bond features.
        for e_idx in range(out.edge_index.size(1)):
            u = int(out.edge_index[0, e_idx].item())
            v = int(out.edge_index[1, e_idx].item())
            if u == r_link or v == r_link:
                for a, val in chosen.items():
                    out[a][e_idx] = val
                out.edge_is_linker[e_idx] = False
        out.is_linker[r_link] = False

    # ---- step C: redirect every r_link to its t_link, drop all r_links ---
    # Build a global remap: drop all r_link_global indices, shift everything
    # above accordingly. r_links are at positions offset+x, all >= n_t, so
    # they sit after the template atoms; the template atoms don't shift.
    redirect = {r_link: t_link for _, t_link, r_link in pairs}
    src = out.edge_index[0].clone()
    dst = out.edge_index[1].clone()
    for r_link, t_link in redirect.items():
        src[src == r_link] = t_link
        dst[dst == r_link] = t_link

    # Build keep_mask and old->new map across all atoms at once.
    total = n_t + sum(n_rs)
    keep_mask = torch.ones(total, dtype=torch.bool)
    for r_link in redirect.keys():
        keep_mask[r_link] = False
    # cumulative shift: each atom's new index = (original index) - (number of
    # dropped atoms with smaller original index)
    drop_counts = torch.zeros(total + 1, dtype=torch.long)
    for r_link in redirect.keys():
        drop_counts[r_link + 1] += 1
    drop_cum = torch.cumsum(drop_counts, dim=0)[:total]
    old_to_new = torch.arange(total) - drop_cum
    out.edge_index = torch.stack([old_to_new[src], old_to_new[dst]])

    for a in NODE_ATTRS:
        out[a] = out[a][keep_mask]
    out.is_linker = out.is_linker[keep_mask]
    out.linker_id = out.linker_id[keep_mask]
    # Clear template atoms' linker_id at the merged positions.
    for _, t_link, _ in pairs:
        out.linker_id[t_link] = 0
    out.num_nodes = int(keep_mask.sum().item())

    # ---- carry forward template's linker_metas for any joints we DIDN'T attach
    consumed = set(lid for lid, _, _ in pairs)
    out.linker_metas = {lid: meta
                        for lid, meta in (template_data.linker_metas or {}).items()
                        if lid not in consumed}

    _ensure_molpla_attrs(out)
    return out


# ---------------------------------------------------------------------------
# convenience: molecule-level operations (pre-mask + attach / detach)
# ---------------------------------------------------------------------------

def attach_rgroup_to_molecule(mol_data: Data,
                              target_atom: int,
                              rgroup: Data,
                              new_linker_id: Optional[int] = None,
                              rgroup_linker_id: Optional[int] = None,
                              rgroup_linker_idx: Optional[int] = None,
                              bond_features: Optional[Dict[str, int]] = None,
                              restore_features: bool = True,
                              ) -> Data:
    """Graft an R-group onto an intact (or partially built) PyG molecule at
    ``target_atom``, pre-masking that atom on the fly.

    Steps:
      1. Take ``mol_data`` and pre-mask ``target_atom``: save its current
         feature snapshot + every incident-bond's features into a fresh
         ``linker_metas`` entry, mark ``is_linker=True``, assign the next
         unused ``linker_id`` (or the explicit ``new_linker_id``).
      2. Call :func:`attach_rgroup` with that masked atom as the template
         joint and the R-group's masked clone as the R-group joint.

    The caller is responsible for choosing a ``target_atom`` with spare
    valence (an aromatic ring C with an implicit H, a primary amine, etc.).
    RDKit sanitisation downstream decides whether the resulting molecule is
    chemically valid; if not, ``pyg_to_mol(..., sanitize=True)`` will raise.

    ``mol_data`` is **not** mutated; an internal clone is masked instead.
    """
    target_atom = int(target_atom)
    if target_atom < 0 or target_atom >= mol_data.num_nodes:
        raise ValueError(f"target_atom {target_atom} out of range")
    # Refuse to overwrite an existing masked joint (would lose linker_metas).
    if bool(mol_data.is_linker[target_atom].item()):
        raise ValueError(
            f"target_atom {target_atom} is already a masked linker; "
            f"use attach_rgroup directly with a linker_id/linker_idx hint")

    # ---- snapshot originals before masking ----
    atom_orig = _atom_feature_snapshot(mol_data, target_atom)
    incident_orig: List[dict] = []
    src, dst = mol_data.edge_index[0], mol_data.edge_index[1]
    for e in range(mol_data.edge_index.size(1)):
        u = int(src[e].item()); v = int(dst[e].item())
        if u == target_atom and v != target_atom and u < v:
            incident_orig.append({
                "neighbour": v,
                "features":  _edge_feature_snapshot(mol_data, e),
            })
        elif v == target_atom and u != target_atom and u < v:
            incident_orig.append({
                "neighbour": u,
                "features":  _edge_feature_snapshot(mol_data, e),
            })

    # ---- clone mol_data and apply pre-mask + linker_id ----
    template = mol_data.clone()
    if not hasattr(template, "linker_id") or template.linker_id is None:
        template.linker_id = torch.zeros(template.num_nodes, dtype=torch.long)
    if not hasattr(template, "linker_metas") or template.linker_metas is None:
        template.linker_metas = {}

    if new_linker_id is None:
        existing_max = int(template.linker_id.max().item()) if template.linker_id.numel() > 0 else 0
        new_linker_id = existing_max + 1
    new_linker_id = int(new_linker_id)
    if new_linker_id <= 0:
        raise ValueError("new_linker_id must be a positive integer")
    if new_linker_id in set(template.linker_id.tolist()):
        raise ValueError(
            f"new_linker_id={new_linker_id} collides with an existing joint id")

    mask_linker_atom(template, target_atom)
    template.linker_id[target_atom] = new_linker_id
    template.linker_metas[new_linker_id] = {
        "atom_features":     atom_orig,
        "incident_features": incident_orig,
        # No cut_bond_features stashed — attach falls back to rgroup's or default.
    }
    _ensure_molpla_attrs(template)

    # ---- attach ----
    return attach_rgroup(
        template, rgroup,
        template_linker_id=new_linker_id,
        rgroup_linker_id=rgroup_linker_id,
        rgroup_linker_idx=rgroup_linker_idx,
        restore_features=restore_features,
        bond_features=bond_features,
    )


# ---------------------------------------------------------------------------
# convenience: explicit aliases for the four "named" advanced ops
# ---------------------------------------------------------------------------

# attach a rgroup with masked linker joint to a TEMPLATE with one or more
# masked linker joints at the designated location:
attach_to_template = attach_rgroup

# detach a rgroup from a TEMPLATE with one or more masked linker joints
# (preserves existing joints; assigns a new linker_id):
detach_from_template = detach_rgroup

# attach a rgroup with masked linker joint to an intact PyG molecule
# (pre-masks the molecule's designated atom first):
attach_to_molecule = attach_rgroup_to_molecule

# detach a rgroup from an intact PyG molecule (pre-masks the designated
# core-side atom on the fly):
detach_from_molecule = detach_rgroup


# ---------------------------------------------------------------------------
# top-level preprocessing: rdkit Mol -> PyG instance (M, template, rgroups)
# ---------------------------------------------------------------------------

def preprocess_molecule(mol_or_smiles,
                        ratio: float = 2.0 / 3.0,
                        do_wash: bool = True,
                        include_ring_substituents: bool = True,
                        decomp_idx: int = 0,
                        store_orig: bool = False,
                        method: Optional[str] = None,
                        **method_kwargs,
                        ) -> Optional[Dict]:
    """One-call preprocessing: rdkit Mol/SMILES -> PyG ``Data`` instance.

    Internally:
      1. ``wash`` the input.
      2. Enumerate putative-core decompositions (RECAP + ring-aware
         fallback when ``include_ring_substituents=True``).
      3. Convert the washed Mol to a PyG ``M``.
      4. Detach all R-groups of the ``decomp_idx``-th decomposition in a
         single :func:`detach_rgroups_multi` pass.

    Parameters
    ----------
    decomp_idx : int
        Which of the enumerated decompositions to materialise. Defaults to
        the first one (largest core, by RECAP's ordering).
    store_orig : bool
        Forwarded to ``detach_rgroups_multi`` — set ``False`` for
        production preprocessing (smaller payload, no round-trip
        capability), ``True`` for verification.

    Returns
    -------
    dict or ``None``
        ``None`` if wash failed. Otherwise a dict with::

            {
                "M":            PyG Data — intact molecule,
                "template":     PyG Data or None — masked-joint template,
                "rgroups":      list[PyG Data] — masked-clone R-groups,
                "smiles":       canonical SMILES of washed M,
                "core_smiles":  decomposition's core fragment SMILES (or None),
                "core_atoms":   tuple of core atom indices in M,
                "n_decomps":    total decompositions enumerated,
            }
    """
    # Local imports to keep the module decoupled from rdkit at module-load
    # time when callers only want graph_ops primitives.
    from rdkit import Chem
    from .decompose import decompose_molecule
    from .mol_features import mol_to_pyg

    if method is not None:
        washed, decs = decompose_molecule(
            mol_or_smiles, do_wash=do_wash, method=method, **method_kwargs)
    else:
        washed, decs = decompose_molecule(
            mol_or_smiles, ratio=ratio, do_wash=do_wash,
            include_ring_substituents=include_ring_substituents)
    if washed is None:
        return None

    M = mol_to_pyg(washed)
    smiles = Chem.MolToSmiles(washed)

    if not decs:
        return {
            "M":           M,
            "template":    None,
            "rgroups":     [],
            "smiles":      smiles,
            "core_smiles": None,
            "core_atoms":  (),
            "n_decomps":   0,
        }

    idx = max(0, min(decomp_idx, len(decs) - 1))
    dec = decs[idx]
    template, rgroups = detach_rgroups_multi(
        M,
        rgroup_infos=[(r.rgroup_atoms, r.core_linker, r.rgroup_linker)
                      for r in dec.rgroups],
        store_orig=store_orig,
    )
    return {
        "M":           M,
        "template":    template,
        "rgroups":     rgroups,
        "smiles":      smiles,
        "core_smiles": dec.core_smiles,
        "core_atoms":  dec.core_atoms,
        "n_decomps":   len(decs),
    }
