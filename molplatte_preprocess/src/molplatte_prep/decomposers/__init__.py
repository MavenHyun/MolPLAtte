"""Decomposition method registry.

MolPallete needs every molecule expressed in MolPLA's **anchored star topology**:
one core plus *k* R-groups, each R-group joined to the core by exactly one shared
linker atom.  Two families of decomposers feed that shape:

*Native anchored* methods produce ``Decomposition`` objects directly:

- ``naveja_recap``  — Naveja putative cores (RECAP children above a size ratio),
  optionally unioned with ring-substituent cores.  This is MolPLA's own method.
- ``bemis_murcko``  — the Murcko scaffold as the core, everything else pendant.

*Multi-cut fragment* methods produce a flat ``FragmentPartition``, which
:mod:`molpallete_prep.anchored_from_partition` re-frames into anchored stars:

- ``macfrag``  — MacFrag (BRICS-like environments + igraph block merging).
- ``synton``   — Synt-On retrosynthetic disconnections.

Why both families: measured on 400 FlavorDB molecules (heavy atoms in [5, 50]),
``naveja_recap`` yields **1.04 R-groups per decomposition** and only 3.7% of its
decompositions have k >= 2 — MolPLA's ``islinked`` subset enumeration and the
core-decoration objective are both degenerate at k = 1.  ``macfrag`` re-framed
through the fragment tree yields **3.34 R-groups per core with 89.7% at k >= 2**
and a mean core of 22.5 heavy atoms, closely matching the ~20.8-heavy-atom cores
MolPLA reports.  See ``../molpallete/docs/molpallete_design.md`` §3.1.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from rdkit import Chem

from .bemis_murcko import decompose_bemis_murcko
from .macfrag import decompose_macfrag
from .naveja_recap import decompose_naveja_recap
from .synton import decompose_synton

__all__ = [
    "ANCHORED_DECOMPOSERS",
    "FRAGMENT_DECOMPOSERS",
    "METHOD_FAMILY",
    "list_methods",
    "family_of",
    "get_anchored_decomposer",
    "get_fragment_decomposer",
]

#: Methods returning ``List[Decomposition]`` (already an anchored star).
ANCHORED_DECOMPOSERS: Dict[str, Callable] = {
    "naveja_recap": decompose_naveja_recap,
    "bemis_murcko": decompose_bemis_murcko,
}

#: Methods returning ``List[FragmentPartition]`` (re-framed downstream).
FRAGMENT_DECOMPOSERS: Dict[str, Callable] = {
    "macfrag": decompose_macfrag,
    "synton": decompose_synton,
}

METHOD_FAMILY: Dict[str, str] = {
    **{m: "anchored" for m in ANCHORED_DECOMPOSERS},
    **{m: "fragment" for m in FRAGMENT_DECOMPOSERS},
}


def list_methods() -> List[str]:
    """Every registered method name, sorted."""
    return sorted(METHOD_FAMILY)


def family_of(method: str) -> str:
    """``"anchored"`` or ``"fragment"`` for *method*.

    Raises
    ------
    ValueError
        If *method* is not registered.
    """
    try:
        return METHOD_FAMILY[method]
    except KeyError:
        raise ValueError(
            f"unknown decomposition method {method!r}; available: {list_methods()}"
        ) from None


def _bind(fn: Callable, name: str, kwargs: dict) -> Callable[[Chem.Mol], list]:
    if not kwargs:
        return fn

    def _bound(mol: Chem.Mol) -> list:
        return fn(mol, **kwargs)

    _bound.__name__ = f"decompose_{name}"
    return _bound


def get_anchored_decomposer(method: str, **kwargs) -> Callable[[Chem.Mol], list]:
    """Return a bound ``mol -> List[Decomposition]`` callable."""
    if method not in ANCHORED_DECOMPOSERS:
        raise ValueError(
            f"{method!r} is not an anchored method; available: "
            f"{sorted(ANCHORED_DECOMPOSERS)}"
        )
    return _bind(ANCHORED_DECOMPOSERS[method], method, kwargs)


def get_fragment_decomposer(method: str, **kwargs) -> Callable[[Chem.Mol], list]:
    """Return a bound ``mol -> List[FragmentPartition]`` callable."""
    if method not in FRAGMENT_DECOMPOSERS:
        raise ValueError(
            f"{method!r} is not a fragment method; available: "
            f"{sorted(FRAGMENT_DECOMPOSERS)}"
        )
    return _bind(FRAGMENT_DECOMPOSERS[method], method, kwargs)
