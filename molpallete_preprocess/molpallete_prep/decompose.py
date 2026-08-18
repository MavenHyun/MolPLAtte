"""Naveja-style putative-core decomposition with MolPLA-style linker joints.

Steps (per molecule M):
  1. wash         - largest fragment, optional charge neutralisation, drop stereo
  2. RECAP        - enumerate all RECAP fragments of M
  3. putative core- keep fragments with NHA >= ratio * NHA(M)
  4. R-groups     - complement atom-set, split into connected components
  5. linker joint - per cut, identify the two heavy atoms on either side of the
                    broken bond (one in the core, one in the R-group)

Ring-aware extension (``include_ring_substituents=True``, default):
  RECAP refuses to recurse when the smaller fragment would be a tiny aromatic
  / single-atom piece, so molecules like caffeine (3 N--CH3 bonds attached to
  a fused aromatic ring system) end up with zero RECAP decompositions even
  though those non-ring bonds are chemically natural cut sites. We
  supplement RECAP with a brute enumeration of every non-ring single bond:
  cut it, check that the cut disconnects the graph into two components, and
  apply the same 2/3 heavy-atom threshold to identify the putative core.
  This is the MolPLA-paper limitation the user asked us to lift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Recap


# -------------------------------------------------- wash --------------------

_NEUTRALISE_PATTERNS = (
    ("[n+;H]",                                "n"),
    ("[N+;!H0]",                              "N"),
    ("[$([O-]);!$([O-][#7])]",                "O"),
    ("[S-;X1]",                               "S"),
    ("[$([N-;X2]S(=O)=O)]",                   "N"),
    ("[$([N-;X2][C,N]=C)]",                   "N"),
    ("[n-]",                                  "[nH]"),
    ("[$([S-]=O)]",                           "S"),
    ("[$([N-]C=O)]",                          "N"),
)
_NEUTRALISE_REACTIONS = None  # lazy


def _neutralise(mol: Chem.Mol) -> Chem.Mol:
    global _NEUTRALISE_REACTIONS
    if _NEUTRALISE_REACTIONS is None:
        _NEUTRALISE_REACTIONS = [
            (Chem.MolFromSmarts(a), Chem.MolFromSmiles(b, sanitize=False))
            for a, b in _NEUTRALISE_PATTERNS
        ]
    cur = mol
    for reactant, product in _NEUTRALISE_REACTIONS:
        while cur.HasSubstructMatch(reactant):
            cur = AllChem.ReplaceSubstructs(cur, reactant, product)[0]
    return cur


def wash(mol_or_smiles, remove_stereo: bool = True) -> Optional[Chem.Mol]:
    """Return a sanitized largest-fragment Mol, or None on failure."""
    if isinstance(mol_or_smiles, str):
        # Largest fragment first (Naveja heuristic for salt removal).
        parts = [p for p in mol_or_smiles.split(".") if p]
        if not parts:
            return None
        parts.sort(key=lambda s: -len(s))
        mol = None
        for p in parts:
            m = Chem.MolFromSmiles(p)
            if m is not None:
                if mol is None or m.GetNumHeavyAtoms() > mol.GetNumHeavyAtoms():
                    mol = m
        if mol is None:
            return None
    else:
        mol = mol_or_smiles
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    mol = _neutralise(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    if remove_stereo:
        Chem.RemoveStereochemistry(mol)
    return mol


# ------------------------------------------------- decomposition ------------

@dataclass(frozen=True)
class RGroupInfo:
    """One R-group attached to a core via one linker bond."""
    rgroup_atoms:   Tuple[int, ...]   # atom indices of the R-group in M
    core_linker:    int               # core-side heavy atom of the cut bond
    rgroup_linker:  int               # R-group-side heavy atom of the cut bond


@dataclass(frozen=True)
class Decomposition:
    """One putative-core + R-groups decomposition of a molecule M."""
    core_smiles:  str
    core_atoms:   Tuple[int, ...]
    rgroups:      Tuple[RGroupInfo, ...]


def _connected_components(mol: Chem.Mol, atoms: Set[int]) -> List[List[int]]:
    components, visited = [], set()
    for start in atoms:
        if start in visited:
            continue
        comp, stack = [], [start]
        while stack:
            a = stack.pop()
            if a in visited:
                continue
            visited.add(a); comp.append(a)
            for n in mol.GetAtomWithIdx(a).GetNeighbors():
                ni = n.GetIdx()
                if ni in atoms and ni not in visited:
                    stack.append(ni)
        components.append(sorted(comp))
    return components


def _strip_dummies(frag: Chem.Mol) -> Chem.Mol:
    em = Chem.EditableMol(frag)
    stars = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(stars, reverse=True):
        em.RemoveAtom(idx)
    out = em.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        pass
    return out


def find_putative_cores(mol: Chem.Mol, ratio: float = 2.0 / 3.0) -> List[Decomposition]:
    """Enumerate Naveja-style putative cores of M.

    A RECAP child fragment qualifies as a putative core if its heavy-atom
    count is >= ``ratio * NHA(M)``. For each qualifying core we map back to
    atom indices in M and identify R-groups + linker joints.

    Multiple cores per molecule are possible ("single molecule — multiple
    scaffolds", per Naveja 2019).
    """
    nha = mol.GetNumHeavyAtoms()
    if nha == 0:
        return []

    try:
        recap = Recap.RecapDecompose(mol)
    except Exception:
        return []
    children = recap.GetAllChildren() if recap is not None else {}

    decomps: List[Decomposition] = []
    seen_core_atoms: Set[Tuple[int, ...]] = set()

    for frag_smiles, node in children.items():
        frag = node.mol
        if frag.GetNumHeavyAtoms() < ratio * nha:
            continue
        core_query = _strip_dummies(frag)
        if core_query.GetNumAtoms() == 0:
            continue

        matches = mol.GetSubstructMatches(core_query, uniquify=True, useChirality=False)
        if not matches:
            continue

        for core_atoms in matches:
            core_set = set(core_atoms)
            r_atoms = set(range(mol.GetNumAtoms())) - core_set
            if not r_atoms:
                continue

            rgroup_infos: List[RGroupInfo] = []
            ok = True
            for comp in _connected_components(mol, r_atoms):
                linker_core = linker_rg = None
                for ai in comp:
                    for nbr in mol.GetAtomWithIdx(ai).GetNeighbors():
                        if nbr.GetIdx() in core_set:
                            linker_rg, linker_core = ai, nbr.GetIdx()
                            break
                    if linker_rg is not None:
                        break
                if linker_rg is None:
                    ok = False
                    break
                rgroup_infos.append(RGroupInfo(
                    rgroup_atoms=tuple(comp),
                    core_linker=linker_core,
                    rgroup_linker=linker_rg,
                ))
            if not ok or not rgroup_infos:
                continue

            key = tuple(sorted(core_set))
            if key in seen_core_atoms:
                continue
            seen_core_atoms.add(key)

            decomps.append(Decomposition(
                core_smiles=frag_smiles,
                core_atoms=key,
                rgroups=tuple(rgroup_infos),
            ))

    return decomps


def _component_after_bond_removal(adj: List[List[Tuple[int, int]]],
                                  n_atoms: int,
                                  bond_idx: int,
                                  start_atom: int) -> List[int]:
    """BFS from ``start_atom`` over ``adj`` skipping ``bond_idx``.

    ``adj[atom_idx]`` is a list of ``(neighbour, bond_idx)`` tuples.
    Returns the reachable-atom list (sorted by visit order).
    """
    visited = [False] * n_atoms
    visited[start_atom] = True
    stack = [start_atom]
    comp = []
    while stack:
        a = stack.pop()
        comp.append(a)
        for nbr, bidx in adj[a]:
            if bidx == bond_idx or visited[nbr]:
                continue
            visited[nbr] = True
            stack.append(nbr)
    return comp


def find_ring_substituent_cores(mol: Chem.Mol, ratio: float = 2.0 / 3.0
                                ) -> List[Decomposition]:
    """Enumerate cuts at every non-ring single bond.

    This is the ring-aware fallback the MolPLA paper / Naveja's RECAP-based
    framework cannot cover. For each non-ring single bond we tentatively
    sever it; if the result has exactly two connected fragments and the
    larger fragment satisfies the 2/3-heavy-atom threshold, we record the
    pair as a putative-core decomposition.

    Bonds skipped:
    - Bonds in any ring (would break a ring open rather than detach an
      attached group).
    - Non-single bonds (cutting C=O, C=C etc. is chemically wrong).

    Symmetry-equivalent matches are deduplicated by the sorted core-atom
    tuple, the same way ``find_putative_cores`` does.
    """
    nha = mol.GetNumHeavyAtoms()
    if nha == 0:
        return []

    n_atoms = mol.GetNumAtoms()
    threshold = ratio * nha

    # Build adjacency once instead of mutating mol for every candidate bond.
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n_atoms)]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bidx = bond.GetIdx()
        adj[i].append((j, bidx))
        adj[j].append((i, bidx))

    decomps: List[Decomposition] = []
    seen: Set[Tuple[int, ...]] = set()

    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            continue

        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # BFS from one side; if it reaches the other side, the bond is in a
        # ring (shouldn't happen given IsInRing check above, but cheap to
        # double-check) and we skip. Otherwise the component is one side.
        comp_a = _component_after_bond_removal(adj, n_atoms, bond.GetIdx(), a)
        if b in comp_a:
            continue
        size_a = len(comp_a)
        size_b = n_atoms - size_a

        if size_a >= size_b:
            core_size = size_a
            core_atoms = comp_a
            rg_atoms = _component_after_bond_removal(adj, n_atoms, bond.GetIdx(), b)
            core_linker, rgroup_linker = a, b
        else:
            core_size = size_b
            rg_atoms = comp_a
            core_atoms = _component_after_bond_removal(adj, n_atoms, bond.GetIdx(), b)
            core_linker, rgroup_linker = b, a

        if core_size < threshold:
            continue

        key = tuple(sorted(core_atoms))
        if key in seen:
            continue
        seen.add(key)

        decomps.append(Decomposition(
            core_smiles="",            # not used downstream; saves a SMILES cost
            core_atoms=key,
            rgroups=(RGroupInfo(
                rgroup_atoms=tuple(sorted(rg_atoms)),
                core_linker=core_linker,
                rgroup_linker=rgroup_linker,
            ),),
        ))

    return decomps


def _bonds_within(mol: Chem.Mol, atom_set: Set[int]) -> List[int]:
    out = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set:
            out.append(bond.GetIdx())
    return out


def decompose_molecule(mol_or_smiles,
                      ratio: float = 2.0 / 3.0,
                      do_wash: bool = True,
                      include_ring_substituents: bool = True,
                      method: Optional[str] = None,
                      **method_kwargs,
                      ) -> Tuple[Optional[Chem.Mol], List[Decomposition]]:
    """End-to-end: wash → enumerate decompositions.

    Backward-compatible default (``method=None``): RECAP + Naveja-ratio
    filter + (default-on) ring-aware fallback. Equivalent to the original
    pre-multi-method behaviour, with ``ratio=2/3`` and
    ``include_ring_substituents=True``.

    Pass ``method=`` to switch fragmentation algorithm; the call is routed
    through :func:`molpallete_prep.decomposers.get_decomposer`. Recognised
    values include ``"recap"``, ``"naveja_recap"``, ``"brics"``,
    ``"rbrics"``, ``"murcko"``, ``"macfrag"``, ``"ring_aware"``. Method-
    specific kwargs flow through as ``**method_kwargs``.

    Returns ``(washed_mol, List[Decomposition])`` or ``(None, [])``.
    """
    mol = wash(mol_or_smiles) if do_wash else (mol_or_smiles if not isinstance(mol_or_smiles, str)
                                               else Chem.MolFromSmiles(mol_or_smiles))
    if mol is None:
        return None, []

    if method is not None:
        # Route through the anchored registry. The fragments paradigm has
        # its own entry point (molpallete_prep.fragments) — not exposed here.
        from .decomposers import get_anchored_decomposer as get_decomposer
        return mol, get_decomposer(method, **method_kwargs)(mol)

    # Legacy path — preserves exact behaviour of pre-multi-method callers.
    decomps = find_putative_cores(mol, ratio=ratio)
    if include_ring_substituents:
        extra = find_ring_substituent_cores(mol, ratio=ratio)
        seen = {d.core_atoms for d in decomps}
        for d in extra:
            if d.core_atoms not in seen:
                decomps.append(d)
                seen.add(d.core_atoms)

    return mol, decomps
