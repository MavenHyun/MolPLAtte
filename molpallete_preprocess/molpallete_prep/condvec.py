"""Condition vectors for the R-group retrieval (RGR) objective.

MolPLA conditions R-group retrieval on a binary functional-group vector of the
*target* R-group: the query is ``g_Phi(q_i (+) c_{R_i})`` where ``q_i`` is the
core-side linker node embedding and ``(+)`` is concatenation (paper Eq. 11).
The paper states ``c_R in [0,1]^87``; the released code allocates 88 and fills it
from ``thermo.functional_groups``.  ``thermo`` is not a dependency here, and its
88 checks are bulk-thermodynamic classes rather than flavor chemistry, so
MolPallete defines its own vector from RDKit.

The paper's ablations make the stakes clear: an all-zero condition (`Cond. None`)
drops retrieval MRR from 0.2616 to 0.0056, and an all-ones condition
(`Cond. All`) to <0.0001.  A degenerate condition is worse than none.

MolPallete supports two modes, selected by ``--condvec-mode``:

``neutral``
    Ligand-only.  85 RDKit ``Chem.Fragments.fr_*`` counters plus 12 SMARTS
    patterns central to flavor and aroma chemistry that ``fr_*`` does not cover
    (pyrazines, pyrroles, di/trisulfides, acetals, isoprene units, ...).
    **97 binary presence bits.**  This is the working default.

``pocket``
    Protein-pocket context.  FlavorDB and COCONUT carry no protein pairings, so
    this ships as a declared interface with a zero-filled implementation --
    :class:`PocketCondVec` is where a taste-receptor or CrossDocked pocket
    encoder plugs in.  Emitting zeros is *deliberately* the `Cond. None` ablation,
    so a pocket run that is silently unfed is visible as collapsed retrieval
    rather than as plausible-looking noise.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Fragments

__all__ = [
    "CondVecEncoder",
    "NeutralCondVec",
    "PocketCondVec",
    "FLAVOR_SMARTS",
    "get_condvec_encoder",
    "CONDVEC_MODES",
]

#: Functional groups central to flavor/aroma chemistry that RDKit's ``fr_*``
#: counters miss.  Each maps a name to a SMARTS pattern.
FLAVOR_SMARTS: Dict[str, str] = {
    # Roasted / nutty / savoury heterocycles -- the Maillard aroma classes.
    "pyrazine":        "c1cnccn1",
    "pyrrole":         "c1cc[nH]c1",
    "pyranone":        "O=C1C=CCO1",
    "thiophene_alkyl": "c1ccsc1-[CX4]",
    # Sulfur volatiles -- alliaceous, tropical, meaty.
    "disulfide":       "[#16X2][#16X2]",
    "trisulfide":      "[#16X2][#16X2][#16X2]",
    "thioester":       "[CX3](=O)[SX2]",
    # Protected carbonyls -- flavour precursors and encapsulation motifs.
    "acetal":          "[CX4H1]([OX2])[OX2]",
    "ketal":           "[CX4]([OX2])([OX2])[#6]",
    # Terpenoid skeleton markers.
    "isoprene_unit":   "CC(=C)C",
    "cyclohexene":     "C1=CCCCC1",
    # Green / fatty note marker: a cis or trans internal olefin.
    "internal_olefin": "[CX3;!$(C=O)]=[CX3;!$(C=O)]",
}


class CondVecEncoder:
    """Interface for condition-vector encoders.

    Implementations must expose a fixed :attr:`dim` and an :meth:`encode` that
    never raises -- a failure returns the zero vector, because a single bad
    R-group must not abort a corpus build.
    """

    #: Stable name recorded in corpus metadata.
    mode: str = "base"

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def encode(self, mol: Chem.Mol, context: Optional[object] = None) -> np.ndarray:
        """Return a ``float32`` vector of length :attr:`dim`."""
        raise NotImplementedError

    def encode_many(
        self, mols: Sequence[Chem.Mol], context: Optional[object] = None
    ) -> np.ndarray:
        return np.vstack([self.encode(m, context) for m in mols]) if mols else \
            np.zeros((0, self.dim), dtype=np.float32)


class NeutralCondVec(CondVecEncoder):
    """Ligand-only functional-group presence vector.

    Binary, not count-valued: MolPLA's vector is a presence indicator, and counts
    would let a single R-group's magnitude dominate the concatenated query.
    """

    mode = "neutral"

    def __init__(self, binary: bool = True) -> None:
        self.binary = binary
        self._fr_names: List[str] = sorted(
            n for n in dir(Fragments) if n.startswith("fr_")
        )
        self._fr_fns = [getattr(Fragments, n) for n in self._fr_names]
        self._smarts_names: List[str] = list(FLAVOR_SMARTS)
        self._smarts = [Chem.MolFromSmarts(FLAVOR_SMARTS[n]) for n in self._smarts_names]
        bad = [n for n, p in zip(self._smarts_names, self._smarts) if p is None]
        if bad:
            raise ValueError(f"FLAVOR_SMARTS failed to parse: {bad}")
        self._dim = len(self._fr_fns) + len(self._smarts)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def feature_names(self) -> List[str]:
        """Human-readable name per bit -- for interpreting retrieval conditions."""
        return self._fr_names + self._smarts_names

    def encode(self, mol: Chem.Mol, context: Optional[object] = None) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        if mol is None:
            return vec
        for i, fn in enumerate(self._fr_fns):
            try:
                count = fn(mol)
            except Exception:
                continue
            if self.binary:
                vec[i] = 1.0 if count > 0 else 0.0
            else:
                vec[i] = float(count)
        offset = len(self._fr_fns)
        for j, patt in enumerate(self._smarts):
            try:
                if mol.HasSubstructMatch(patt):
                    vec[offset + j] = 1.0
            except Exception:
                continue
        return vec


class PocketCondVec(CondVecEncoder):
    """Protein-pocket condition vector -- declared interface, unfed implementation.

    MolPLA's own stated future work is "R-group retrieval conditioned by protein
    pockets" (paper section 4.3).  MolPallete reserves the slot and fixes the
    contract so the training side needs no change when a pocket source arrives:

    - ``context`` is the pocket handle (a PDB path, a pocket graph, or a
      precomputed embedding); ``None`` means ligand-only.
    - :meth:`encode` must return a fixed-width ``float32`` vector regardless.

    Until a source is wired, this returns zeros, which reproduces the paper's
    `Cond. None` ablation exactly -- retrieval collapses visibly instead of
    training on plausible-looking noise.

    Candidate sources, neither yet built:

    - taste / olfactory receptors (T1R2-T1R3, T2R family, ORs) -- would need
      structure acquisition plus pockets, and flavor molecules have no measured
      receptor pairings, so pairing would be docking-derived.
    - CrossDocked2020 (on disk at ``~/datasets/crossdocked2020``) -- real
      pocket-ligand pairs, but drug targets rather than taste receptors.
    """

    mode = "pocket"

    def __init__(self, dim: int = 97, encoder: Optional[CondVecEncoder] = None) -> None:
        self._dim = dim
        self._encoder = encoder

    @property
    def dim(self) -> int:
        return self._dim if self._encoder is None else self._encoder.dim

    @property
    def is_fed(self) -> bool:
        """``False`` while no pocket encoder is attached."""
        return self._encoder is not None

    def encode(self, mol: Chem.Mol, context: Optional[object] = None) -> np.ndarray:
        if self._encoder is not None:
            return self._encoder.encode(mol, context)
        return np.zeros(self._dim, dtype=np.float32)


CONDVEC_MODES = ("neutral", "pocket")


def get_condvec_encoder(mode: str = "neutral", **kwargs) -> CondVecEncoder:
    """Build the condition-vector encoder named *mode*."""
    if mode == "neutral":
        return NeutralCondVec(**kwargs)
    if mode == "pocket":
        neutral_dim = NeutralCondVec().dim
        kwargs.setdefault("dim", neutral_dim)
        return PocketCondVec(**kwargs)
    raise ValueError(f"unknown condvec mode {mode!r}; available: {list(CONDVEC_MODES)}")
