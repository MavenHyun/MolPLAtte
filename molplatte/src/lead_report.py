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


def render_gallery(results, input_smiles: str, out_path: Path,
                   flavor_condition: Sequence[str] = (),
                   per_row: int = 4, note: str = "",
                   reference: Optional[Dict[str, Optional[float]]] = None
                   ) -> Optional[Path]:
    """One section per (decomposition, slot): the core, then its products.

    Written as a PDF when the suffix says so, otherwise PNG. Returns None rather
    than raising if the drawing stack is unavailable -- a missing picture should
    not cost the caller their table.
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

    def panel(smiles_list, legends, title):
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        keep = [(m, l) for m, l in zip(mols, legends) if m is not None]
        if not keep:
            return None
        img = Draw.MolsToGridImage([m for m, _ in keep], molsPerRow=per_row,
                                   subImgSize=(300, 240),
                                   legends=[l for _, l in keep],
                                   returnPNG=False)
        fig_w = 3.2 * min(per_row, len(keep))
        rows = (len(keep) + per_row - 1) // per_row
        fig, ax = plt.subplots(figsize=(fig_w, 2.7 * rows + 0.6))
        ax.imshow(np.asarray(img)); ax.axis("off")
        ax.set_title(title, fontsize=11, loc="left")
        fig.tight_layout()
        return fig

    figs = []
    spec = ""
    if reference:
        spec = "   ".join(f"{k} {v:.2f}" for k, v in reference.items()
                          if v is not None)
    head = panel([input_smiles],
                 [f"INPUT  {input_smiles}" + (f"\n{spec}" if spec else "")],
                 f"Input compound   flavor={list(flavor_condition) or 'none'}"
                 + (f"\n{note}" if note else ""))
    if head is not None:
        figs.append(head)
    for slot in results:
        prods = [(s.product, s) for s in slot.suggestions if s.product]
        if not prods:
            continue
        legends = [f"#{s.rank}  {s.smiles}\nscore {s.score:+.2f}"
                   f"{'  [novel]' if s.is_novel else ''}" for _, s in prods]
        f = panel([p for p, _ in prods], legends,
                  f"decomposition {slot.decomp_index}, slot {slot.slot_index}"
                  f"   core {slot.core_smiles or '?'}"
                  f"   replaced {slot.original_rgroup}")
        if f is not None:
            figs.append(f)
    if not figs:
        return None
    if out_path.suffix.lower() == ".pdf":
        with PdfPages(out_path) as pdf:
            for f in figs:
                pdf.savefig(f); plt.close(f)
    else:
        for i, f in enumerate(figs):
            p = out_path if i == 0 else out_path.with_name(
                f"{out_path.stem}_{i}{out_path.suffix}")
            f.savefig(p, dpi=150); plt.close(f)
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
                                      reference=reference)

    return LeadOptimizationReport(
        input_smiles=smiles, flavor_condition=labels, reference=reference,
        table=table, failures=failures,
        compounds=[p for p in table["product"].tolist()] if len(table) else [],
        gallery=gallery_path, results=results,
        pocket_used=pocket_condition is not None,
        pocket_changed_ranking=changed)
