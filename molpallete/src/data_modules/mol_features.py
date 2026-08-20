"""RDKit Mol <-> torch_geometric Data with MolPLA's exact feature schema.

Feature indices match those used in MolPLA's `mol_to_nx_molpla`
(github.com/dmis-lab/MolPLA). The extra "MASK" index per attribute is
reserved for nodes/edges flagged as masked linker joints.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from rdkit import Chem
from torch_geometric.data import Data


class MolPalleteData(Data):
    """``torch_geometric.data.Data`` subclass that knows how to batch the
    MolPLA-specific attributes correctly.

    Custom behaviour:

    - ``linker_atom`` is a 1-D ``LongTensor`` of variable length (one entry
      per masked joint, sorted by ``linker_id``). Default PyG batching would
      treat its values as plain integers; we override ``__inc__`` so that
      ``Batch.from_data_list`` automatically offsets each graph's
      ``linker_atom`` by that graph's ``num_nodes`` — meaning ``batch.x[batch.linker_atom]``
      indexes the correct atoms in the batched node table.
    - ``linker_metas`` (a Python ``dict[int, dict]`` keyed by linker_id) is
      kept as a plain Python attribute. PyG's default Batch behaviour merges
      it heuristically and does not always round-trip; downstream code that
      needs the metas should call ``batch.to_data_list()`` to recover them
      per-graph.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "linker_atom":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


RDKIT_FEATURES = {
    # ---- node attributes ----
    "atomic_num":      list(range(0, 128)),
    "formal_charge":   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "chiral_tag":      list(Chem.rdchem.ChiralType.names.values()),
    "hybridization":   list(Chem.rdchem.HybridizationType.names.values()),
    # GetTotalNumHs, not GetNumExplicitHs. The latter counts only hydrogens
    # written explicitly in the SMILES ([nH]), which is 94% zeros -- entropy
    # 0.23 against 1.22 for the total. MolPLA/MolDAM both carry the explicit
    # count, so the encoder effectively had no hydrogen information.
    # Reconstruction is unaffected: pyg_to_mol feeds this to SetNumExplicitHs,
    # and RDKit only adds implicit Hs up to the default valence, so setting
    # explicit = total saturates to the same molecule. Verified on 2,500
    # molecules and on [nH]/charged subsets: 99.96% round-trip either way.
    "total_num_hs":    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "is_aromatic":     [False, True],
    # Ring membership is NOT recoverable by message passing (cycle detection
    # needs more than 3-5 hops), unlike degree which a GNN can count from the
    # adjacency. Both OGB and PyG carry it. 62/38 split, entropy 0.66.
    "is_in_ring":      [False, True],
    # ---- edge attributes ----
    "bond_type":        list(Chem.rdchem.BondType.names.values()),
    "edge_is_aromatic": [False, True],
    "is_conjugated":    [False, True],
    "bond_dir":         list(Chem.rdchem.BondDir.names.values()),
    "bond_stereo":      list(Chem.rdchem.BondStereo.names.values()),
    "edge_is_in_ring":  [False, True],
}

# Reserve the index past the last valid one as the "MASK" for that attribute.
MASK_VALUES = {k: len(v) for k, v in RDKIT_FEATURES.items()}

NODE_ATTRS = ["atomic_num", "formal_charge", "chiral_tag",
              "hybridization", "total_num_hs", "is_aromatic", "is_in_ring"]
EDGE_ATTRS = ["bond_type", "edge_is_aromatic", "is_conjugated",
              "bond_dir", "bond_stereo", "edge_is_in_ring"]


def _idx(name: str, value) -> int:
    table = RDKIT_FEATURES[name]
    try:
        return table.index(value)
    except ValueError:
        # Out-of-vocab — clamp to last valid index; MolPLA also has this fallback
        # for very rare features such as exotic chiral tags. Mask-index is
        # reserved for linker joints, so we don't return it here.
        return len(table) - 1


def mol_to_pyg(mol: Chem.Mol, linker_atoms: Optional[Iterable[int]] = None) -> Data:
    """Convert an RDKit Mol to a PyG ``Data`` using MolPLA's feature schema.

    Parameters
    ----------
    mol : Chem.Mol
    linker_atoms : iterable of int, optional
        Atom indices to flag as ``is_linker=True`` (i.e. masked linker joints).
        Edges incident on a linker atom are flagged ``edge_is_linker=True``.

    Notes
    -----
    The linker flag is **only** a metadata bit; the attribute values themselves
    are left intact. Call ``graph_ops.mask_linker_atom`` to additionally
    overwrite them with the reserved MASK indices.
    """
    linker_set = set(linker_atoms or [])

    # ---------------- node tensors ----------------
    atomic_num      = []
    formal_charge   = []
    chiral_tag      = []
    hybridization   = []
    total_num_hs    = []
    is_aromatic     = []
    is_in_ring      = []
    is_linker_node  = []

    for atom in mol.GetAtoms():
        atomic_num.append(_idx("atomic_num", atom.GetAtomicNum()))
        formal_charge.append(_idx("formal_charge", atom.GetFormalCharge()))
        chiral_tag.append(_idx("chiral_tag", atom.GetChiralTag()))
        hybridization.append(_idx("hybridization", atom.GetHybridization()))
        total_num_hs.append(_idx("total_num_hs", atom.GetTotalNumHs()))
        is_aromatic.append(_idx("is_aromatic", atom.GetIsAromatic()))
        is_in_ring.append(_idx("is_in_ring", atom.IsInRing()))
        is_linker_node.append(atom.GetIdx() in linker_set)

    # ---------------- edge tensors (undirected -> stored bidirectional) ----------------
    src, dst         = [], []
    bond_type        = []
    edge_is_aromatic = []
    is_conjugated    = []
    edge_is_linker   = []
    bond_dir         = []
    bond_stereo      = []
    edge_is_in_ring  = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt   = _idx("bond_type", bond.GetBondType())
        ear  = _idx("edge_is_aromatic", bond.GetIsAromatic())
        icj  = _idx("is_conjugated", bond.GetIsConjugated())
        bdr  = _idx("bond_dir", bond.GetBondDir())
        bst  = _idx("bond_stereo", bond.GetStereo())
        bir  = _idx("edge_is_in_ring", bond.IsInRing())
        link = (i in linker_set) or (j in linker_set)
        for u, v in ((i, j), (j, i)):
            src.append(u); dst.append(v)
            bond_type.append(bt)
            edge_is_aromatic.append(ear)
            is_conjugated.append(icj)
            edge_is_linker.append(link)
            bond_dir.append(bdr)
            bond_stereo.append(bst)
            edge_is_in_ring.append(bir)

    n_atoms = mol.GetNumAtoms()
    data = MolPalleteData(
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        num_nodes=n_atoms,
    )
    # All 13 attributes have vocab sizes ≤128, fit in uint8 (0–255).
    # int8 would NOT work — atomic_num's MASK index is 128 which
    # overflows int8's [-128, 127] range and wraps to -128.  Upcast
    # to long only at nn.Embedding lookup time (VanillaGNN.forward).
    data.atomic_num       = torch.tensor(atomic_num,       dtype=torch.uint8)
    data.formal_charge    = torch.tensor(formal_charge,    dtype=torch.uint8)
    data.chiral_tag       = torch.tensor(chiral_tag,       dtype=torch.uint8)
    data.hybridization    = torch.tensor(hybridization,    dtype=torch.uint8)
    data.total_num_hs     = torch.tensor(total_num_hs,     dtype=torch.uint8)
    data.is_aromatic      = torch.tensor(is_aromatic,      dtype=torch.uint8)
    data.is_in_ring       = torch.tensor(is_in_ring,       dtype=torch.uint8)
    data.is_linker        = torch.tensor(is_linker_node,   dtype=torch.bool)

    data.bond_type        = torch.tensor(bond_type,        dtype=torch.uint8)
    data.edge_is_aromatic = torch.tensor(edge_is_aromatic, dtype=torch.uint8)
    data.is_conjugated    = torch.tensor(is_conjugated,    dtype=torch.uint8)
    data.bond_dir         = torch.tensor(bond_dir,         dtype=torch.uint8)
    data.bond_stereo      = torch.tensor(bond_stereo,      dtype=torch.uint8)
    data.edge_is_in_ring  = torch.tensor(edge_is_in_ring,  dtype=torch.uint8)
    data.edge_is_linker   = torch.tensor(edge_is_linker,   dtype=torch.bool)

    # linker_id stays long — it's used as a scatter / lookup index.
    # edge_index stays long — PyG requires it.
    data.linker_id    = torch.zeros(n_atoms, dtype=torch.long)
    data.linker_atom  = torch.zeros(0, dtype=torch.long)
    data.linker_metas = {}

    return data


def pyg_to_mol(data: Data, sanitize: bool = True) -> Chem.Mol:
    """Reverse mapping for verification. Returns an editable RDKit Mol.

    Atoms or bonds flagged with MASK values (e.g. atomic_num==128) become
    placeholder dummy atoms / single bonds — useful for canonical SMILES
    checks but not full round-trip fidelity.
    """
    em = Chem.RWMol()
    fc_table  = RDKIT_FEATURES["formal_charge"]
    ct_table  = RDKIT_FEATURES["chiral_tag"]
    hyb_table = RDKIT_FEATURES["hybridization"]
    bt_table  = RDKIT_FEATURES["bond_type"]
    bdr_table = RDKIT_FEATURES["bond_dir"]
    bst_table = RDKIT_FEATURES["bond_stereo"]

    for i in range(data.num_nodes):
        z = int(data.atomic_num[i])
        if z == MASK_VALUES["atomic_num"]:
            z = 0  # dummy atom
        atom = Chem.Atom(z)
        fc_i = int(data.formal_charge[i])
        if fc_i != MASK_VALUES["formal_charge"]:
            atom.SetFormalCharge(fc_table[fc_i])
        ct_i = int(data.chiral_tag[i])
        if ct_i != MASK_VALUES["chiral_tag"]:
            atom.SetChiralTag(ct_table[ct_i])
        hb_i = int(data.hybridization[i])
        if hb_i != MASK_VALUES["hybridization"]:
            atom.SetHybridization(hyb_table[hb_i])
        # total_num_hs -> SetNumExplicitHs is deliberate, see RDKIT_FEATURES.
        # Ring flags are not restored: RDKit derives ring membership from the
        # bond graph, so setting them would be redundant and can conflict.
        nh_i = int(data.total_num_hs[i])
        if nh_i != MASK_VALUES["total_num_hs"]:
            atom.SetNumExplicitHs(nh_i)
        ar_i = int(data.is_aromatic[i])
        if ar_i != MASK_VALUES["is_aromatic"]:
            atom.SetIsAromatic(bool(ar_i))
        em.AddAtom(atom)

    seen = set()
    for e in range(data.edge_index.size(1)):
        i = int(data.edge_index[0, e])
        j = int(data.edge_index[1, e])
        if (i, j) in seen or (j, i) in seen:
            continue
        seen.add((i, j))
        bt_i = int(data.bond_type[e])
        bt = bt_table[bt_i] if bt_i != MASK_VALUES["bond_type"] else Chem.rdchem.BondType.SINGLE
        em.AddBond(i, j, bt)
        bond = em.GetBondBetweenAtoms(i, j)
        bdr_i = int(data.bond_dir[e])
        if bdr_i != MASK_VALUES["bond_dir"]:
            bond.SetBondDir(bdr_table[bdr_i])
        bst_i = int(data.bond_stereo[e])
        if bst_i != MASK_VALUES["bond_stereo"]:
            bond.SetStereo(bst_table[bst_i])

    mol = em.GetMol()
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
    return mol


# ---------------------------------------------------------------------------
# portable serialization — pickle a Data/MolPalleteData as a plain dict so the
# resulting pickle bytes don't reference ``molpallete.mol_features.MolPalleteData``
# as a class qualname. The dict only contains tensors + Python primitives,
# so the pickle is loadable anywhere torch + torch_geometric are available.
# ---------------------------------------------------------------------------

_PORTABLE_TAG_DATA    = "MolPalleteData"
_PORTABLE_TAG_DATA_V2 = "MolPalleteData_v2"


def _tensor_to_numpy(x):
    import numpy as _np
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _numpy_to_tensor(x):
    import numpy as _np
    if isinstance(x, _np.ndarray):
        return torch.from_numpy(x)
    return x


def data_to_portable(data) -> dict:
    """Convert a PyG Data / MolPalleteData to a plain dict.

    Output schema (v2):

    * ``__t``         → the marker ``"MolPalleteData_v2"``.
    * ``__num_nodes`` → explicit int, so empty / no-edge graphs round-trip.
    * every key in ``data.keys()`` → its value with ``torch.Tensor`` fields
      converted to ``numpy.ndarray``. Numpy round-trips through pickle
      without invoking ``torch.load``/``_load_from_bytes``, so unpickling
      is fork-safe under multi-worker DataLoaders.
    * (``linker_metas`` is included as a plain dict if present.)

    Legacy v1 stores (torch.Tensor payloads, ``__t == "MolPalleteData"``) are
    still readable by :func:`portable_to_data`.
    """
    out: dict = {
        "__t":          _PORTABLE_TAG_DATA_V2,
        "__num_nodes":  int(data.num_nodes),
    }
    for key in data.keys():
        out[key] = _tensor_to_numpy(data[key])
    if "linker_metas" not in out and hasattr(data, "linker_metas"):
        out["linker_metas"] = data.linker_metas
    return out


def portable_to_data(d: dict, cls=None):
    """Inverse of :func:`data_to_portable`.

    Accepts both v1 (torch tensor payload) and v2 (numpy payload) records
    transparently. ``cls`` defaults to :class:`MolPalleteData`.
    """
    if cls is None:
        cls = MolPalleteData
    num_nodes = d.get("__num_nodes")
    out = cls()
    for k, v in d.items():
        if k in ("__t", "__num_nodes"):
            continue
        out[k] = _numpy_to_tensor(v)
    if num_nodes is not None:
        out.num_nodes = num_nodes
    return out


def is_portable_data(obj) -> bool:
    """True iff ``obj`` looks like a dict produced by :func:`data_to_portable`.
    Accepts both v1 (torch tensor payload) and v2 (numpy payload) markers."""
    return isinstance(obj, dict) and obj.get("__t") in (
        _PORTABLE_TAG_DATA, _PORTABLE_TAG_DATA_V2)
