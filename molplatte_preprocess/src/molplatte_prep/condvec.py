"""Condition vectors for the R-group retrieval (RGR) objective.

MolPLA conditions R-group retrieval on a binary functional-group vector of the
*target* R-group: the query is ``g_Phi(q_i (+) c_{R_i})`` where ``q_i`` is the
core-side linker node embedding and ``(+)`` is concatenation (paper Eq. 11).
The paper states ``c_R in [0,1]^87``; the released code allocates 88 and fills it
from ``thermo.functional_groups``.  ``thermo`` is not a dependency here, and its
88 checks are bulk-thermodynamic classes rather than flavor chemistry, so
MolPLAtte defines its own vector from RDKit.

The paper's ablations make the stakes clear: an all-zero condition (`Cond. None`)
drops retrieval MRR from 0.2616 to 0.0056, and an all-ones condition
(`Cond. All`) to <0.0001.  A degenerate condition is worse than none.

MolPLAtte supports two modes, selected by ``--condvec-mode``:

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
    "FlavorCondVec",
    "TwoPartCondVec",
    "FLAVOR_LABELS",
    "load_flavor_tables",
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

    #: Numpy dtype the corpus should store this encoder's output as.
    #: uint8 is right for the binary flavor bits and the RDKit fragment counts,
    #: and it is a quarter of the size. It is CATASTROPHIC for a real-valued
    #: embedding: ESM-2 activations are negative floats, so -6.668 wraps to 250
    #: and everything in (-1, 1) truncates to 0. The result is an array of the
    #: right shape and dtype containing none of the original information, which
    #: nothing downstream can detect. Encoders that emit real values must say so.
    storage_dtype = np.uint8

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
    pockets" (paper section 4.3).  MolPLAtte reserves the slot and fixes the
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


#: The 24 flavor bits. Folded from FlavorDB's 715 raw descriptors (253 of them
#: singletons, 67% of mass on sweet/sweet-like); this set covers 94.3% of
#: genuine non-imputed assignments. Order is FIXED -- it is the wire format.
FLAVOR_LABELS: Sequence[str] = (
    "sweet", "bitter", "sour", "salty", "umami",
    "fruity", "green", "floral", "fatty", "woody", "spicy", "roasted",
    "sulfurous", "earthy", "nutty", "herbal", "medicinal", "citrus",
    "dairy", "alcoholic", "meaty", "minty",
    "odorless",   # positive claim: cannot volatilise / no perceptible odor
    "unknown",    # positive claim: no information. NOT the same as odorless.
)
_FLAVOR_IDX = {l: i for i, l in enumerate(FLAVOR_LABELS)}
_ODORLESS_I = _FLAVOR_IDX["odorless"]
_UNKNOWN_I = _FLAVOR_IDX["unknown"]

#: The five taste primaries. A taste receptor sits in solution, so these are
#: compatible with `odorless` -- sucrose is both, and so is glutamate.
TASTE_LABELS = frozenset(FLAVOR_LABELS[:5])

#: The seventeen odour classes. These CONTRADICT `odorless`, which is a
#: positive claim that nothing is smelled.
ODOUR_LABELS = frozenset(FLAVOR_LABELS[5:22])

#: Molecular weight above which a compound cannot reach an olfactory receptor.
#: Real odorants (FlavorDB, non-imputed) have median MW 170 with 8.5% above 350;
#: COCONUT has median 433 with 71.2% above. This is a claim about VOLATILITY and
#: therefore about odor only -- taste receptors sit in solution and steviosides
#: are intensely sweet at MW ~800, so a sugar-bearing heavy compound is NOT
#: assigned odorless-and-nothing-else.
ODORLESS_MW_CUTOFF = 350.0


class StoredPocketCondVec(CondVecEncoder):
    """Pocket half read from the record, not computed here.

    The pocket encoder is a frozen protein language model that needs a GPU and
    the full chain sequence, neither of which belongs inside a 96-way
    multiprocessing decomposition. So embedding happens once up front
    (``embed_pockets_esm.py``) and the vector rides along in the record's meta;
    this class only validates and hands it over.

    The vector is stored RAW, at the language model's own width -- 1280 for
    ESM-2 650M. It is not reduced here, because the reduction has to be learned:
    ``nnet_modules.components.PocketConditioning`` projects it down inside the
    model, where gradients can decide what to keep. Storing a reduced vector
    would freeze that choice at preprocessing time.

    A record with no pocket gets zeros, and zeros are meaningful downstream:
    ``PocketConditioning`` masks an all-zero pocket half to exactly zero, so
    ligand-only pretraining stays bit-identical to a flavor-only run.
    """

    mode = "stored_pocket"
    storage_dtype = np.float32

    #: Meta key the reader writes the embedding to.
    KEY = "pocket_embedding"

    def __init__(self, dim: int) -> None:
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, mol: Chem.Mol, context: Optional[object] = None) -> np.ndarray:
        ctx = context if isinstance(context, dict) else {}
        raw = ctx.get(self.KEY)
        if raw is None:
            return np.zeros(self._dim, dtype=np.float32)
        vec = np.asarray(raw, dtype=np.float32).ravel()
        if vec.shape[0] != self._dim:
            # Silently truncating or padding here would produce a corpus whose
            # pocket half is misaligned with the encoder that made it, and
            # nothing downstream could tell.
            raise ValueError(
                f"stored pocket embedding has width {vec.shape[0]}, "
                f"expected {self._dim}"
            )
        if not np.isfinite(vec).all():
            raise ValueError("stored pocket embedding contains NaN or inf")
        return vec


class FlavorCondVec(CondVecEncoder):
    """24 sparse flavor bits, sourced from measurement -- never from structure.

    This is the half of the condition vector that must NOT be a function of the
    molecular graph. The project's original 97-bit RDKit fragment condvec was
    exactly that, and it leaked: it was computed from the intact molecule G,
    which contains the R-group retrieval has to predict. Removing it moved
    rank1_distinct from 680 to 5,851 and coverage from 11.4% to 100%.

    So bits are set from three sources, in priority order, and every one of them
    is exogenous to the graph:

    1. MEASURED  -- FlavorDB and any other curated sensory database, joined by
       InChIKey. ~21,387 compounds in coconut-flavordb.
    2. MINED     -- LLM annotations at the `documented` evidence tier ONLY, i.e.
       literature recall with a named source (96.4% cite Good Scents, BitterDB
       or FEMA). `structural` and `close_analog` tiers are DELIBERATELY excluded:
       both are inferred from the graph and would reintroduce the leak.
       ~1,927 compounds.
    3. PHYSICS   -- `odorless` for compounds too heavy to volatilise, which is a
       statement about physics rather than about missing data. ~208,978
       compounds. Applies to odor only; heavy sugar-bearing compounds are left
       unknown because they can still be tasted.

    Anything else gets the `unknown` bit. An ALL-ZERO vector is never emitted:
    zero would mean "no flavor", and 94% of the corpus being silently labelled
    flavourless would teach the model to separate FlavorDB-sourced from
    COCONUT-sourced molecules -- source provenance wearing flavor's clothes,
    which is the same failure as the original condvec.
    """

    mode = "flavor"

    def __init__(self,
                 measured: Optional[Dict[str, Sequence[str]]] = None,
                 mined: Optional[Dict[str, Sequence[str]]] = None,
                 mw_cutoff: float = ODORLESS_MW_CUTOFF) -> None:
        self._measured = measured or {}
        self._mined = mined or {}
        self._mw_cutoff = float(mw_cutoff)

    @property
    def dim(self) -> int:
        return len(FLAVOR_LABELS)

    def _set(self, labels: Sequence[str]) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        seen = set()
        for l in labels:
            name = str(l).strip().lower()
            i = _FLAVOR_IDX.get(name)
            if i is not None:
                v[i] = 1.0
                seen.add(name)

        # `odorless` asserts that nothing is smelled, so it cannot hold
        # alongside an odour class. 145 rows of flavor_measured.jsonl (1.3%)
        # carry both -- a merge artifact from combining sources without
        # resolving conflicts, e.g. FDB72 ['medicinal', 'odorless', 'woody'].
        # The odour descriptor is the specific claim and wins.
        #
        # Taste is NOT affected: a taste receptor sits in solution, so
        # `odorless` + `sweet` is the correct description of sucrose.
        if seen & ODOUR_LABELS:
            v[_ODORLESS_I] = 0.0

        if not v.any():
            v[_UNKNOWN_I] = 1.0
        return v

    def encode(self, mol: Chem.Mol, context: Optional[object] = None) -> np.ndarray:
        """``context`` is the record dict (needs ``inchikey`` / ``mol_id`` / ``mw``)."""
        ctx = context if isinstance(context, dict) else {}
        key = (ctx.get("inchikey") or "").strip()
        mid = (ctx.get("mol_id") or "").strip()

        for table in (self._measured, self._mined):
            for k in (key, mid):
                if k and k in table:
                    return self._set(table[k])

        # physics: too heavy to be an odorant. Sugar-bearing heavies can still
        # TASTE, so they stay unknown rather than being called odorless.
        mw = ctx.get("mw")
        if mw is None and mol is not None:
            from rdkit.Chem import Descriptors
            try: mw = Descriptors.MolWt(mol)
            except Exception: mw = None
        if mw is not None and float(mw) > self._mw_cutoff and not ctx.get("contains_sugar"):
            return self._set(("odorless",))
        return self._set(())          # -> unknown bit


class TwoPartCondVec(CondVecEncoder):
    """``[flavor bits | pocket embedding]`` -- the two halves have opposite roles.

    flavor  sparse, categorical, from measurement/literature/physics. Present
            during pretraining wherever known.
    pocket  dense, continuous, from a pocket graph encoder. ZERO throughout
            pretraining on flavordb+coconut (no protein pairings exist), filled
            only during pocket finetuning on tastepocket.

    The pocket half is leak-free by construction: a receptor structure is not a
    function of the ligand. That is the property every flavor-derived signal
    failed, and it is why the pocket stage rests on firmer ground than the
    flavor stage despite having only 59 taste-strict complexes.
    """

    mode = "two_part"

    def __init__(self, flavor: "FlavorCondVec", pocket: "PocketCondVec") -> None:
        self.flavor = flavor
        self.pocket = pocket

    @property
    def storage_dtype(self):
        """Widen to the more permissive of the two halves.

        A real-valued pocket half forces float32 for the whole vector; storing
        the pair as uint8 to save space on the 24 flavor bits would wrap the
        1280 pocket floats to garbage.
        """
        for half in (self.flavor, self.pocket):
            if np.dtype(half.storage_dtype).kind == "f":
                return np.float32
        return np.uint8

    @property
    def dim(self) -> int:
        return self.flavor.dim + self.pocket.dim

    @property
    def is_pocket_fed(self) -> bool:
        return self.pocket.is_fed

    def encode(self, mol: Chem.Mol, context: Optional[object] = None) -> np.ndarray:
        return np.concatenate([self.flavor.encode(mol, context),
                               self.pocket.encode(mol, context)]).astype(np.float32)


def load_flavor_tables(measured_path=None, mined_path=None):
    """``(measured, mined)`` label tables keyed by InChIKey and by mol_id."""
    import json
    meas: Dict[str, Sequence[str]] = {}
    mined: Dict[str, Sequence[str]] = {}
    for path, table, doc_only in ((measured_path, meas, False), (mined_path, mined, True)):
        if not path:
            continue
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try: r = json.loads(line)
            except Exception: continue
            # mined labels are usable ONLY at the documented tier
            if doc_only and r.get("evidence") != "documented":
                continue
            labs = [l for l in (r.get("labels") or []) if l != "unknown"]
            if not labs:
                continue
            for k in ((r.get("inchikey") or "").strip(), (r.get("id") or "").strip()):
                if k:
                    table[k] = labs
    return meas, mined


#: Content version of the flavor condition vector. Bump whenever the MEANING of
#: the bits changes, so two corpora with the same mode+dim are still marked
#: incomparable. v1 corpora are unusable for conditioning: mol_context was read
#: from a record that had no `meta` yet, so no label table was ever consulted and
#: all 24 bits collapsed to a single MW>350 indicator (fixed 2026-09).
CONDVEC_VERSION = 3

CONDVEC_MODES = ("neutral", "pocket", "flavor", "two_part")


def get_condvec_encoder(mode: str = "neutral", **kwargs) -> CondVecEncoder:
    """Build the condition-vector encoder named *mode*."""
    if mode == "neutral":
        return NeutralCondVec(**kwargs)
    if mode == "pocket":
        neutral_dim = NeutralCondVec().dim
        kwargs.setdefault("dim", neutral_dim)
        return PocketCondVec(**kwargs)
    if mode == "flavor":
        return FlavorCondVec(**kwargs)
    if mode == "two_part":
        pocket_dim = kwargs.pop("pocket_dim", 0)
        return TwoPartCondVec(FlavorCondVec(**kwargs), PocketCondVec(dim=pocket_dim))
    raise ValueError(f"unknown condvec mode {mode!r}; available: {list(CONDVEC_MODES)}")
