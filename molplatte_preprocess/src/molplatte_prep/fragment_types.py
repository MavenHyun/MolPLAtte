"""Type 2 (Fragment Partition) paradigm — data containers.

A :class:`FragmentPartition` is a flat partition of a molecule M into k
connected sub-graphs ("fragments") with no privileged "core". Every atom
of M belongs to exactly one fragment; the bonds of M that cross the
partition are recorded as :class:`CutBond` instances.

Contrast with :mod:`molplatte_prep.anchored` (Type 1), which represents M as a
**star-shaped** decomposition: one core hub + k R-group spokes, each
spoke connecting to the core via exactly one masked linker joint. The
fragment paradigm relaxes the star constraint, so chain
(``A--B--C--D``), tree, and arbitrary partition topologies are all
representable.

Masked-linker semantics in the fragments paradigm
-------------------------------------------------

For each cut bond ``C_k = (u, v)`` with ``u`` in fragment ``F_a`` and
``v`` in fragment ``F_b``:

* ``F_a`` gets a **fresh masked clone** ``v_stub`` (a new atom) bonded to
  ``F_a``'s ``u`` via a masked edge. ``linker_id[v_stub] = k``.
* ``F_b`` gets a **fresh masked clone** ``u_stub`` bonded to ``F_b``'s
  ``v``. ``linker_id[u_stub] = k``.

In other words, the cut bond is "punched in two" symmetrically: each
side carries a masked stand-in for the atom on the other side. Both
clones share the same ``linker_id`` so :func:`attach_fragments` can pair
them. The cut bond's original features live in
``linker_metas[k]["cut_bond_features"]`` (carried on either side).

This generalises the anchored model in two ways:

1. **Topology** — both endpoints get a clone, not just the
   "R-group-side" endpoint, so neither fragment is privileged.
2. **Geminal cuts** — if atom ``u`` participates in multiple cut bonds
   (e.g. ``u--v`` and ``u--w``), then ``u``'s fragment gets one masked
   clone *per* cut. Each clone has a distinct ``linker_id``, so the
   scalar-per-atom ``linker_id`` invariant is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CutBond:
    """One cut bond in a fragment partition.

    Attributes
    ----------
    linker_id : int
        Positive integer uniquely identifying this cut (across the
        partition). Used to pair the two masked clones during attach.
    u_atom_in_M : int
        Atom index of one cut endpoint in the parent molecule M.
    v_atom_in_M : int
        Atom index of the other cut endpoint in M.
    u_frag : int
        Fragment index containing ``u_atom_in_M``.
    v_frag : int
        Fragment index containing ``v_atom_in_M``.
    """
    linker_id: int
    u_atom_in_M: int
    v_atom_in_M: int
    u_frag: int
    v_frag: int


@dataclass(frozen=True)
class FragmentInfo:
    """One connected fragment in a partition.

    Attributes
    ----------
    atoms_in_M : tuple[int, ...]
        Atom indices from M belonging to this fragment. The indices are
        in their order of appearance in M; they don't carry any
        ordering semantics beyond identity.
    """
    atoms_in_M: Tuple[int, ...]


@dataclass(frozen=True)
class FragmentPartition:
    """A flat partition of M into k fragments connected by ``cut_bonds``.

    Invariants enforced by :func:`validate_partition`:

    * The atom sets of ``fragments`` are disjoint and cover ``range(N)``
      where ``N == mol.GetNumAtoms()``.
    * Each ``CutBond.linker_id`` is unique and positive.
    * Each cut bond's endpoints land in different fragments (otherwise
      cutting it wouldn't disconnect anything) and the underlying bond
      exists in M.
    * Cutting all ``cut_bonds`` from M produces exactly ``len(fragments)``
      connected components matching ``fragments``.

    Attributes
    ----------
    fragments : tuple[FragmentInfo, ...]
        The k connected partitions of M's atoms (k ≥ 1).
    cut_bonds : tuple[CutBond, ...]
        The bonds whose removal partitions M into ``fragments``. Empty
        when ``len(fragments) == 1`` (the trivial whole-molecule
        "partition").
    """
    fragments: Tuple[FragmentInfo, ...]
    cut_bonds: Tuple[CutBond, ...]

    @property
    def n_fragments(self) -> int:
        return len(self.fragments)

    @property
    def n_cuts(self) -> int:
        return len(self.cut_bonds)


def validate_partition(partition: FragmentPartition, n_atoms: int) -> None:
    """Validate the structural invariants of a partition. Raise on failure."""
    seen_atoms = set()
    for i, frag in enumerate(partition.fragments):
        s = set(frag.atoms_in_M)
        if not s:
            raise ValueError(f"fragment {i} has no atoms")
        if s & seen_atoms:
            raise ValueError(
                f"fragment {i} overlaps an earlier fragment at "
                f"atoms {s & seen_atoms}")
        seen_atoms |= s
    if seen_atoms != set(range(n_atoms)):
        missing = set(range(n_atoms)) - seen_atoms
        extra = seen_atoms - set(range(n_atoms))
        raise ValueError(
            f"fragments must cover [0, {n_atoms}); missing={sorted(missing)} "
            f"extra={sorted(extra)}")

    seen_ids = set()
    for c in partition.cut_bonds:
        if c.linker_id <= 0:
            raise ValueError(f"cut bond linker_id must be positive: {c}")
        if c.linker_id in seen_ids:
            raise ValueError(f"duplicate linker_id={c.linker_id} in cut_bonds")
        seen_ids.add(c.linker_id)
        if c.u_frag == c.v_frag:
            raise ValueError(
                f"cut bond {c} has both endpoints in fragment {c.u_frag}")
        for frag_idx, atom_idx in [(c.u_frag, c.u_atom_in_M),
                                    (c.v_frag, c.v_atom_in_M)]:
            if atom_idx not in partition.fragments[frag_idx].atoms_in_M:
                raise ValueError(
                    f"cut bond endpoint atom {atom_idx} not in fragment "
                    f"{frag_idx}'s atom set")


# ---------------------------------------------------------------------------
# portable serialization — same intent as molplatte_prep.mol_features.data_to_portable
# ---------------------------------------------------------------------------

_PORTABLE_TAG_PARTITION = "FragmentPartition"


def fragment_partition_to_portable(p: FragmentPartition) -> dict:
    """Serialize a :class:`FragmentPartition` to Python primitives so the
    pickle bytes don't reference ``molplatte_prep.fragments.data_types`` (or
    wherever this module lives).
    """
    return {
        "__t":       _PORTABLE_TAG_PARTITION,
        "fragments": [list(fi.atoms_in_M) for fi in p.fragments],
        # 5 ints per cut bond; positional for compactness:
        # [linker_id, u_atom_in_M, v_atom_in_M, u_frag, v_frag]
        "cut_bonds": [
            [int(cb.linker_id), int(cb.u_atom_in_M), int(cb.v_atom_in_M),
             int(cb.u_frag),    int(cb.v_frag)]
            for cb in p.cut_bonds
        ],
    }


def portable_to_fragment_partition(d: dict) -> FragmentPartition:
    return FragmentPartition(
        fragments=tuple(
            FragmentInfo(atoms_in_M=tuple(atoms))
            for atoms in d["fragments"]
        ),
        cut_bonds=tuple(
            CutBond(linker_id=cb[0], u_atom_in_M=cb[1], v_atom_in_M=cb[2],
                    u_frag=cb[3], v_frag=cb[4])
            for cb in d["cut_bonds"]
        ),
    )


def is_portable_partition(obj) -> bool:
    return isinstance(obj, dict) and obj.get("__t") == _PORTABLE_TAG_PARTITION
