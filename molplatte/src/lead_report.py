"""Turn LeadOptimizer output into a scored, visualised, exportable report.

`LeadOptimizer.optimize` answers "which R-groups belong at this joint". This
answers the question a chemist actually asks: show me the molecules, score them,
and give me something I can read and hand to someone else.

Three things here are load-bearing and easy to get wrong:

* **Deduplication.** `optimize` returns one result per (decomposition, slot), so
  a compound with four decompositions and three slots each can propose the same
  product a dozen times. Counting those as a dozen compounds overstates what the
  model produced, so products are keyed on canonical SMILES, the best-scoring
  occurrence wins, and the slots that proposed it are recorded.

* **The score.** `retrieval_score` is the logQ-corrected value that actually
  ranks: `sim/tau + log p(k)`. It is NOT a similarity and not in [-1, 1] -- it
  sits near +45 for this checkpoint. Sorting the table by anything else
  reorders it away from what the model believes.

* **Assembly can fail, and that is reported rather than hidden.** A retrieved
  R-group whose predicted joint chemistry is invalid for this core yields no
  product. Those rows go to `.failures` with their reason instead of being
  dropped, because a table of 40 compounds that silently began as 60 misstates
  the hit rate.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Draw, QED, RDConfig

RDLogger.DisableLog("rdApp.*")

# Same resolution lead_optimization.py uses: the preprocessing package is a
# sibling checkout, not an installed dependency.
_PREP = Path(__file__).resolve().parents[2] / "molplatte_preprocess" / "src"
if str(_PREP) not in sys.path:
    sys.path.insert(0, str(_PREP))

from molplatte_prep.condvec import FLAVOR_LABELS  # noqa: E402

#: The strings `flavor_condition` accepts. 5 taste + 17 odour + odorless +
#: unknown, matching the 24 condvec bits the corpus stores.
PERMISSIBLE_FLAVORS: tuple = tuple(FLAVOR_LABELS)
TASTE_FLAVORS: tuple = tuple(FLAVOR_LABELS[:5])
ODOUR_FLAVORS: tuple = tuple(FLAVOR_LABELS[5:22])

_NP_MODEL = None


def _contrib(subdir: str, module: str):
    path = os.path.join(RDConfig.RDContribDir, subdir)
    if path not in sys.path:
        sys.path.append(path)
    return __import__(module)


def _sa_score(mol) -> Optional[float]:
    try:
        return float(_contrib("SA_Score", "sascorer").calculateScore(mol))
    except Exception:  # noqa: BLE001
        return None


def _np_score(mol) -> Optional[float]:
    global _NP_MODEL
    try:
        npscorer = _contrib("NP_Score", "npscorer")
        if _NP_MODEL is None:
            _NP_MODEL = npscorer.readNPModel()
        return float(npscorer.scoreMol(mol, _NP_MODEL))
    except Exception:  # noqa: BLE001
        return None


def as_mol(compound):
    """Accept a SMILES string or an RDKit Mol; return (mol, canonical smiles)."""
    if isinstance(compound, Chem.Mol):
        mol = compound
    elif isinstance(compound, str):
        mol = Chem.MolFromSmiles(compound)
        if mol is None:
            raise ValueError(f"not parseable as SMILES: {compound!r}")
    else:
        raise TypeError(
            f"input_compound must be a SMILES string or rdkit.Chem.Mol, "
            f"got {type(compound).__name__}"
        )
    return mol, Chem.MolToSmiles(mol)


def check_flavors(flavor_condition: Optional[Sequence[str]]) -> List[str]:
    """Validate against the 24 permissible strings, naming the bad ones."""
    labels = list(flavor_condition or ())
    bad = [f for f in labels if f not in PERMISSIBLE_FLAVORS]
    if bad:
        raise ValueError(
            f"unknown flavor label(s) {bad}. Permissible: "
            f"{', '.join(PERMISSIBLE_FLAVORS)}"
        )
    return labels


def molecule_scores(smiles: str) -> Dict[str, Optional[float]]:
    """MW / logP / QED / SA / NP for one product SMILES."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return dict(MW=None, logP=None, QED=None, SAScore=None, NPScore=None)
    try:
        qed = float(QED.qed(mol))
    except Exception:  # noqa: BLE001
        qed = None
    return dict(
        MW=float(Descriptors.MolWt(mol)),
        logP=float(Crippen.MolLogP(mol)),
        QED=qed,
        SAScore=_sa_score(mol),
        NPScore=_np_score(mol),
    )


@dataclass
class LeadOptimizationReport:
    """Everything one call produced."""

    input_smiles: str
    flavor_condition: List[str]
    table: "object"                      # pandas.DataFrame, deduped products
    failures: "object"                   # pandas.DataFrame, unassembled
    #: The INPUT compound's own specs, on the same scale as the products'.
    reference: Dict[str, Optional[float]] = field(default_factory=dict)
    compounds: List[str] = field(default_factory=list)
    gallery: Optional[Path] = None
    results: list = field(default_factory=list)
    pocket_used: bool = False
    pocket_changed_ranking: Optional[bool] = None

    def reference_row(self):
        """The input compound as a one-row DataFrame, for display beside the
        products. Kept OUT of `.table` deliberately: it has no retrieval_score,
        so a sorted table would have to place it arbitrarily."""
        import pandas as pd

        return pd.DataFrame([{"product": self.input_smiles,
                              "rgroup": "(input)", "retrieval_score": None,
                              **{k: self.reference.get(k) for k in SPEC_KEYS},
                              "is_input": True}])

    def __repr__(self) -> str:  # noqa: D105
        n = len(self.compounds)
        return (f"<LeadOptimizationReport {self.input_smiles!r} "
                f"flavor={self.flavor_condition} "
                f"{n} compound{'s' if n != 1 else ''}"
                f"{', pocket INERT' if self.pocket_changed_ranking is False else ''}>")


#: Properties carried for both the input and every product, so the table can
#: show the change a substitution made rather than only its absolute value.
SPEC_KEYS = ("MW", "logP", "QED", "SAScore", "NPScore")


def build_tables(results, input_smiles: str,
                 reference: Optional[Dict[str, Optional[float]]] = None):
    """Deduplicated product table plus the assembly failures.

    ``reference`` is the INPUT compound's specs. When given, each product also
    carries ``d<prop>`` = product - input, which is the number a chemist
    actually reads: an absolute QED of 0.74 means little, +0.10 against the
    starting compound means something.
    """
    import pandas as pd

    best: Dict[str, dict] = {}
    fails: List[dict] = []
    for slot in results:
        for s in slot.suggestions:
            where = f"d{slot.decomp_index}/s{slot.slot_index}"
            if not s.product:
                fails.append(dict(
                    decomposition=slot.decomp_index, slot=slot.slot_index,
                    rank=s.rank, rgroup=s.smiles,
                    retrieval_score=float(s.score),
                    reason=s.product_error or "no product"))
                continue
            row = best.get(s.product)
            if row is None or s.score > row["retrieval_score"]:
                keep = dict(
                    product=s.product, rgroup=s.smiles,
                    retrieval_score=float(s.score),
                    rank=s.rank, decomposition=slot.decomp_index,
                    slot=slot.slot_index, core=slot.core_smiles,
                    replaced=slot.original_rgroup,
                    corpus_count=s.corpus_count, is_novel=bool(s.is_novel),
                    aromaticity_kept=s.aromaticity_kept,
                    is_input=(s.product == input_smiles),
                    **molecule_scores(s.product))
                if reference:
                    for k in SPEC_KEYS:
                        a, b = keep.get(k), reference.get(k)
                        keep[f"d{k}"] = (None if a is None or b is None
                                         else float(a) - float(b))
                keep["found_in_slots"] = (row or {}).get("found_in_slots", set()) | {where}
                best[s.product] = keep
            else:
                row["found_in_slots"].add(where)

    rows = sorted(best.values(), key=lambda r: -r["retrieval_score"])
    for r in rows:
        slots = sorted(r.pop("found_in_slots"))
        r["n_slots"] = len(slots)
        r["found_in_slots"] = ",".join(slots)
    cols = ["product", "rgroup", "retrieval_score", "MW", "logP", "QED",
            "SAScore", "NPScore"]
    if reference:
        cols += [f"d{k}" for k in SPEC_KEYS]
    cols += ["is_novel", "corpus_count", "aromaticity_kept", "is_input",
             "n_slots", "found_in_slots", "core", "replaced",
             "decomposition", "slot", "rank"]
    table = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    failures = pd.DataFrame(fails) if fails else pd.DataFrame(
        columns=["decomposition", "slot", "rank", "rgroup",
                 "retrieval_score", "reason"])
    return table, failures


def _highlight_png(mol, core_atoms, rgroup_atoms, size=(560, 420)):
    """Parent depiction with the CORE and the R-GROUP BEING REPLACED coloured.

    This is the picture that explains a decomposition: which part is held
    fixed, and which part the model is proposing replacements for. Bonds
    internal to each set are coloured too, otherwise only the atoms read as
    highlighted and the split is hard to see at a glance.
    """
    from rdkit.Chem.Draw import rdMolDraw2D

    core = set(core_atoms or ())
    rgrp = set(rgroup_atoms or ())
    CORE_C, RG_C = (0.68, 0.85, 0.90), (1.00, 0.72, 0.60)
    atom_cols = {i: CORE_C for i in core}
    atom_cols.update({i: RG_C for i in rgrp})

    bonds, bond_cols = [], {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        col = CORE_C if (i in core and j in core) else (
            RG_C if (i in rgrp and j in rgrp) else None)
        if col is not None:
            bonds.append(b.GetIdx())
            bond_cols[b.GetIdx()] = col

    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.drawOptions().useBWAtomPalette()
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=list(atom_cols), highlightAtomColors=atom_cols,
        highlightBonds=bonds, highlightBondColors=bond_cols)
    d.FinishDrawing()
    import io

    from PIL import Image
    return Image.open(io.BytesIO(d.GetDrawingText()))


def _score_caption(row) -> str:
    """Every score, on two lines, with the delta against the input."""
    def f(key, fmt="{:.2f}"):
        v = row.get(key)
        return "n/a" if v is None else fmt.format(v)

    def d(key, fmt="{:+.2f}"):
        v = row.get("d" + key)
        return "" if v is None else f" ({fmt.format(v)})"

    return (f"MW {f('MW','{:.1f}')}{d('MW','{:+.1f}')}   "
            f"logP {f('logP')}{d('logP')}   QED {f('QED')}{d('QED')}\n"
            f"SA {f('SAScore')}{d('SAScore')}   "
            f"NP {f('NPScore')}{d('NPScore')}   "
            f"count {int(row.get('corpus_count') or 0):,}"
            f"{'   NOVEL' if row.get('is_novel') else ''}")


def render_gallery(results, input_smiles: str, out_path: Path,
                   flavor_condition: Sequence[str] = (),
                   per_row: int = 3, note: str = "",
                   reference: Optional[Dict[str, Optional[float]]] = None,
                   table=None) -> Optional[Path]:
    """One section per (decomposition, slot).

    Each opens with the parent molecule showing which atoms are the CORE and
    which are the R-group being replaced, then the products with every score
    beneath them. `table` supplies the scores; without it the captions fall
    back to the retrieval score alone.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:  # noqa: BLE001
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parent = Chem.MolFromSmiles(input_smiles)
    by_product = {}
    if table is not None and len(table):
        by_product = {r["product"]: r for r in table.to_dict("records")}

    figs = []

    # ---- title page: the input and its own specs
    fig = plt.figure(figsize=(11, 6.2))
    ax = fig.add_subplot(111); ax.axis("off")
    if parent is not None:
        ax.imshow(np.asarray(_highlight_png(parent, [], [], (760, 380))),
                  extent=(0, 1, 0.18, 0.92), aspect="auto")
    spec = ""
    if reference:
        spec = "     ".join(f"{k} {v:.2f}" for k, v in reference.items()
                            if v is not None)
    ax.set_title(f"Lead optimization: {input_smiles}\n"
                 f"flavour = {list(flavor_condition) or 'none'}",
                 fontsize=13, loc="left")
    ax.text(0, 0.10, "INPUT  " + spec, fontsize=10, family="monospace")
    if note:
        ax.text(0, 0.03, note, fontsize=10, color="crimson", weight="bold")
    figs.append(fig)

    # ---- one section per slot
    for slot in results:
        prods = [s for s in slot.suggestions if s.product]
        if not prods:
            continue
        rows = (len(prods) + per_row - 1) // per_row
        # Generous vertical room: each product carries a two-line score caption
        # BELOW its axes, which collides with the next row's title if the grid
        # is packed tight.
        fig = plt.figure(figsize=(4.8 * per_row, 4.9 * rows + 3.0))
        gs = fig.add_gridspec(rows + 1, per_row,
                              height_ratios=[1.6] + [1] * rows,
                              hspace=0.55, wspace=0.10)

        head = fig.add_subplot(gs[0, :]); head.axis("off")
        if parent is not None:
            head.imshow(np.asarray(_highlight_png(
                parent, slot.core_atoms, slot.rgroup_atoms, (900, 380))),
                extent=(0, 1, 0, 1), aspect="auto")
        head.set_title(
            f"decomposition {slot.decomp_index}, slot {slot.slot_index}"
            f"      core (blue) {slot.core_smiles or '?'}"
            f"      replacing (orange) {slot.original_rgroup}",
            fontsize=11, loc="left")

        for k, s in enumerate(prods):
            ax = fig.add_subplot(gs[1 + k // per_row, k % per_row])
            ax.axis("off")
            m = Chem.MolFromSmiles(s.product)
            if m is not None:
                ax.imshow(np.asarray(_highlight_png(m, [], [], (520, 380))))
            row = by_product.get(s.product, {})
            ax.set_title(f"#{s.rank}  {s.smiles}   score {s.score:+.2f}",
                         fontsize=9, loc="left")
            ax.text(0, -0.10, _score_caption(row) if row else "",
                    transform=ax.transAxes, fontsize=8, family="monospace",
                    va="top")
        figs.append(fig)

    if not figs:
        return None
    if out_path.suffix.lower() == ".pdf":
        with PdfPages(out_path) as pdf:
            for f in figs:
                pdf.savefig(f, bbox_inches="tight"); plt.close(f)
    else:
        for i, f in enumerate(figs):
            q = out_path if i == 0 else out_path.with_name(
                f"{out_path.stem}_{i}{out_path.suffix}")
            f.savefig(q, dpi=150, bbox_inches="tight"); plt.close(f)
    return out_path


def _ranking_of(results) -> tuple:
    """A hashable fingerprint of the full ranking, for comparing two runs."""
    return tuple((sl.decomp_index, sl.slot_index, s.rank, s.hash)
                 for sl in results for s in sl.suggestions)


def run_lead_optimization(input_compound, lead_optimizer, flavor_condition,
                          top_k: int = 10, *, pocket_condition=None,
                          gallery: Optional[str | Path] = None,
                          max_decompositions: int = 4,
                          verify_pocket: bool = True) -> LeadOptimizationReport:
    """Retrieve, assemble, score, visualise.

    ``pocket_condition`` is a 1280-d ESM-2 pocket embedding, e.g. from
    ``pocket_embedding_from_structure``.

    WHEN A POCKET IS SUPPLIED IT IS VERIFIED, NOT TRUSTED. The shipped
    checkpoint's pocket reduction is zero-initialised and was never trained, so
    a pocket contributes exactly nothing -- measured null at 0, 1,056 and
    168,096 trainable parameters (docs/step3_pocket_capacity_2026-09-09.md).
    This runs retrieval twice, with and without, and states plainly whether the
    pocket changed the ranking. An API whose central argument silently does
    nothing is worse than one that says so.
    """
    mol, smiles = as_mol(input_compound)
    labels = check_flavors(flavor_condition)

    results = lead_optimizer.optimize(
        smiles, flavor=labels, pocket=pocket_condition, top_k=top_k,
        max_decompositions=max_decompositions, assemble=True)

    changed = None
    if pocket_condition is not None and verify_pocket:
        control = lead_optimizer.optimize(
            smiles, flavor=labels, pocket=None, top_k=top_k,
            max_decompositions=max_decompositions, assemble=False)
        # assemble= differs between the two calls but does not touch ranking,
        # so the fingerprints are directly comparable.
        changed = _ranking_of(control) != _ranking_of(results)
        if not changed:
            import warnings
            warnings.warn(
                "POCKET HAD NO EFFECT: the ranking is identical with and "
                "without it. This checkpoint's pocket reduction is untrained "
                "(zero-init), so the pocket contributes exactly zero. Treat "
                "this result as flavour-conditioned only.",
                RuntimeWarning, stacklevel=2)

    reference = molecule_scores(smiles)
    table, failures = build_tables(results, smiles, reference)
    gallery_path = None
    if gallery:
        note = ""
        if changed is False:
            note = "POCKET SUPPLIED BUT INERT -- ranking identical without it"
        gallery_path = render_gallery(results, smiles, Path(gallery),
                                      flavor_condition=labels, note=note,
                                      reference=reference, table=table)

    return LeadOptimizationReport(
        input_smiles=smiles, flavor_condition=labels, reference=reference,
        table=table, failures=failures,
        compounds=[p for p in table["product"].tolist()] if len(table) else [],
        gallery=gallery_path, results=results,
        pocket_used=pocket_condition is not None,
        pocket_changed_ranking=changed)
