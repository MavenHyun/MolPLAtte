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
    #: RMSD of the redocked crystal ligand against its own pose. Under ~2 A
    #: means the docking setup reproduces known truth; None means no docking.
    redock_rmsd: Optional[float] = None
    docked_receptor: Optional[str] = None
    #: Directory of top docked poses as SDF, one per compound. A score without
    #: its pose cannot be inspected.
    pose_dir: Optional[str] = None

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
                f"{', pocket INERT' if self.pocket_changed_ranking is False else ''}"
                f"{f', redock {self.redock_rmsd:.2f}A' if self.redock_rmsd is not None else ''}>")


#: Properties carried for both the input and every product, so the table can
#: show the change a substitution made rather than only its absolute value.
SPEC_KEYS = ("MW", "logP", "QED", "SAScore", "NPScore")


def build_tables(results, input_smiles: str,
                 reference: Optional[Dict[str, Optional[float]]] = None,
                 vina: Optional[Dict[str, Optional[float]]] = None):
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
                if vina is not None:
                    keep["vina_score"] = vina.get(s.product)
                    ref_v = (reference or {}).get("vina_score")
                    if keep["vina_score"] is not None and ref_v is not None:
                        keep["dvina"] = float(keep["vina_score"]) - float(ref_v)
                    else:
                        keep["dvina"] = None
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
    if vina is not None:
        cols += ["vina_score", "dvina"]
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

    lines = [f"MW   {f('MW','{:.1f}')}{d('MW','{:+.1f}')}"
             f"      logP {f('logP')}{d('logP')}"
             f"      QED  {f('QED')}{d('QED')}",
             f"SA   {f('SAScore')}{d('SAScore')}"
             f"      NP   {f('NPScore')}{d('NPScore')}"
             f"      corpus count {int(row.get('corpus_count') or 0):,}"]
    if row.get("vina_score") is not None:
        lines.append(f"vina {row['vina_score']:+.2f} kcal/mol"
                     + (f"  ({row['dvina']:+.2f} vs input)"
                        if row.get("dvina") is not None else ""))
    return "\n".join(lines)


#: Interpreter that can `import pymol`. PyMOL pulls Qt5/X11 and conda would put
#: its numpy underneath a pip-installed torch, so it lives in its OWN env and is
#: driven as a subprocess. Absent, the matplotlib renderer below is used --
#: PyMOL is optional, not required.
PYMOL_PYTHON = os.environ.get(
    "MOLPLATTE_PYMOL_PYTHON",
    str(Path.home() / "miniconda3" / "envs" / "molplatte-viz" / "bin" / "python"))
_PYMOL_SCRIPT = Path(__file__).resolve().parent / "scripts" / "render_pose_pymol.py"


def pymol_available() -> bool:
    return Path(PYMOL_PYTHON).is_file() and _PYMOL_SCRIPT.is_file()


def _pose_png_pymol(sdf_path, receptor_pdb, out_png) -> bool:
    """Ray-trace one pose with PyMOL. False if it did not produce an image."""
    import subprocess

    try:
        subprocess.run(
            [PYMOL_PYTHON, str(_PYMOL_SCRIPT), str(receptor_pdb),
             str(sdf_path), str(out_png)],
            capture_output=True, timeout=300, check=False)
    except Exception:  # noqa: BLE001 - fall back rather than lose the page
        return False
    return Path(out_png).is_file() and Path(out_png).stat().st_size > 0


def _pose_page_from_image(png_path, size=(9.0, 6.9)):
    """Wrap a rendered PNG in a figure so it can join the PDF pages."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.imshow(mpimg.imread(str(png_path)))
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0)
    return fig


def _pose_png(sdf_path, receptor_atoms=None, size=(6.6, 5.0), contact=6.0):
    """3D depiction of a docked pose with its contacting pocket atoms.

    Matplotlib rather than PyMOL/py3Dmol: py3Dmol renders to HTML/JS, which a
    PDF page cannot embed, and PyMOL is not installed.

    The pose is ROTATED INTO ITS OWN PRINCIPAL FRAME before drawing. A docked
    ligand sits at an arbitrary orientation in crystal coordinates, and a fixed
    camera catches most of them edge-on -- a flat aromatic system then reads as
    a line. Aligning the two largest principal axes to the screen plane shows
    the molecule face-on every time, without hand-tuning a view per compound.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rdkit import Chem

    supp = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mol = next(iter(supp), None)
    if mol is None or mol.GetNumConformers() == 0:
        return None
    P = mol.GetConformer().GetPositions()
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy) < 3:
        return None

    C = P[heavy]
    centre = C.mean(0)
    # Principal axes of the HEAVY atoms only; hydrogens would tilt the frame.
    _, _, Vt = np.linalg.svd(C - centre, full_matrices=False)
    P = (P - centre) @ Vt.T
    C = P[heavy]

    ELEM = {6: ("#333333", 90), 7: ("#2c5aa0", 120), 8: ("#c0392b", 120),
            16: ("#b8860b", 150), 9: ("#27ae60", 100), 17: ("#27ae60", 130),
            35: ("#8b4513", 160)}

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")

    if receptor_atoms is not None and len(receptor_atoms):
        R = (np.asarray(receptor_atoms) - centre) @ Vt.T
        d = np.linalg.norm(R[:, None, :] - C[None, :, :], axis=2).min(1)
        near = R[d < contact]
        if len(near):
            ax.scatter(near[:, 0], near[:, 1], near[:, 2], s=26, c="#7d9ec0",
                       alpha=0.42, linewidths=0, depthshade=False)

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 1 or \
           mol.GetAtomWithIdx(j).GetAtomicNum() == 1:
            continue
        ax.plot(*zip(P[i], P[j]), c="#2b2b2b", lw=3.4, solid_capstyle="round",
                zorder=3)
    for idx in heavy:
        col, sz = ELEM.get(mol.GetAtomWithIdx(idx).GetAtomicNum(), ("#7f5fa0", 110))
        ax.scatter(*P[idx], s=sz, c=col, depthshade=False, linewidths=0, zorder=4)

    # Frame on the ligand, not on the ligand+pocket cloud, and keep the two
    # in-plane axes equal so the depiction is not stretched.
    # np.ptp(): ndarray.ptp() was removed in numpy 2.0
    half = max(np.ptp(C[:, 0]), np.ptp(C[:, 1])) / 2 + 1.8
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_zlim(C[:, 2].min() - 2.5, C[:, 2].max() + 2.5)
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 0.55))
    except Exception:  # noqa: BLE001 - older matplotlib
        pass
    # After PCA alignment the molecule's plane IS the xy plane, so a steep
    # elevation looks nearly down onto it: face-on, with just enough tilt left
    # to read depth. `ax.dist` is deprecated, so the crop comes from filling
    # the axes rather than moving the camera.
    ax.view_init(elev=66, azim=-72)
    ax.set_position([0.0, -0.10, 1.0, 1.06])
    return fig


def render_gallery(results, input_smiles: str, out_path: Path,
                   flavor_condition: Sequence[str] = (),
                   per_row: int = 3, note: str = "",
                   reference: Optional[Dict[str, Optional[float]]] = None,
                   table=None, poses: Optional[Dict[str, Path]] = None,
                   receptor_atoms=None, receptor_pdb=None) -> Optional[Path]:
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

    # ---- one section per slot: ONE COMPOUND PER ROW
    # A grid packed the caption under each cell, where a two-line score block
    # and the next row's title ran into each other. One row per compound gives
    # the text a full page width and cannot collide.
    for slot in results:
        prods = [s for s in slot.suggestions if s.product]
        if not prods:
            continue
        n = len(prods)
        fig = plt.figure(figsize=(11.5, 2.4 * n + 3.4))
        gs = fig.add_gridspec(n + 1, 2, width_ratios=[1.5, 2.0],
                              height_ratios=[2.3] + [1] * n,
                              hspace=0.42, wspace=0.02)

        head = fig.add_subplot(gs[0, :]); head.axis("off")
        if parent is not None:
            head.imshow(np.asarray(_highlight_png(
                parent, slot.core_atoms, slot.rgroup_atoms, (1150, 420))),
                extent=(0, 1, 0, 1), aspect="auto")
        head.set_title(
            f"decomposition {slot.decomp_index}, slot {slot.slot_index}"
            f"      core (blue) {slot.core_smiles or '?'}"
            f"      replacing (orange) {slot.original_rgroup}",
            fontsize=10, loc="left")

        for k, s_ in enumerate(prods):
            axm = fig.add_subplot(gs[1 + k, 0]); axm.axis("off")
            m = Chem.MolFromSmiles(s_.product)
            if m is not None:
                axm.imshow(np.asarray(_highlight_png(m, [], [], (720, 500))))
            axt = fig.add_subplot(gs[1 + k, 1]); axt.axis("off")
            row = {}
            if table is not None and len(table):
                hit = table[table["product"] == s_.product]
                if len(hit):
                    row = hit.iloc[0].to_dict()
            axt.text(0.0, 0.96,
                     f"#{s_.rank}   {s_.smiles}"
                     f"{'   [novel]' if s_.is_novel else ''}",
                     transform=axt.transAxes, fontsize=10, va="top",
                     family="monospace", weight="bold")
            axt.text(0.0, 0.70, f"score {s_.score:+.2f}",
                     transform=axt.transAxes, fontsize=9, va="top",
                     family="monospace")
            axt.text(0.0, 0.50, _score_caption(row) if row else "",
                     transform=axt.transAxes, fontsize=9, va="top",
                     family="monospace")
            axt.text(0.0, 0.06, s_.product, transform=axt.transAxes,
                     fontsize=7.5, va="top", family="monospace", color="#666666",
                     wrap=True)
        figs.append(fig)

    # ---- docked poses, best-scoring first
    if poses:
        order = (list(table["product"]) if table is not None and len(table)
                 else list(poses))
        shown = 0
        for prod in order:
            sdf = poses.get(prod)
            if not sdf or not Path(sdf).is_file() or shown >= 6:
                continue
            fig, how = None, ""
            if receptor_pdb and pymol_available():
                png = Path(sdf).with_suffix(".render.png")
                if _pose_png_pymol(sdf, receptor_pdb, png):
                    fig = _pose_page_from_image(png)
                    how = ("PyMOL: pocket surface, ligand sticks, "
                           "polar contacts dashed")
            if fig is None:                       # PyMOL absent or failed
                fig = _pose_png(sdf, receptor_atoms)
                how = "pocket atoms within 6 A in blue"
            if fig is None:
                continue
            row = {}
            if table is not None and len(table):
                hit = table[table["product"] == prod]
                if len(hit):
                    row = hit.iloc[0].to_dict()
            v = row.get("vina_score")
            dv = row.get("dvina")
            fig.suptitle(
                f"docked pose   {prod}\n"
                + (f"vina {v:+.2f} kcal/mol" if v is not None else "")
                + (f"   ({dv:+.2f} vs input)" if dv is not None else "")
                + f"   {how}",
                fontsize=9, x=0.02, ha="left")
            figs.append(fig)
            shown += 1

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
                          verify_pocket: bool = True,
                          receptor: Optional[str | Path] = None,
                          dock: bool = False,
                          exhaustiveness: int = 8,
                          ligand_resname: Optional[str] = None,
                          ligand_resname_smiles: Optional[str] = None,
                          pose_dir: Optional[str | Path] = None
                          ) -> LeadOptimizationReport:
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

            # Two very different reasons the ranking did not move, and saying
            # the wrong one is worse than saying nothing: a zero adapter CANNOT
            # do anything, while a trained adapter that changes nothing is the
            # measured weakness of pocket conditioning on this input.
            # nn.ModuleDict is not attribute-accessible by key -- getattr
            # returns None and the message silently reports "untrained".
            nnet = getattr(lead_optimizer.model, "nnet", None)
            pc = (nnet["pocket_conditioning"]
                  if nnet is not None and "pocket_conditioning" in nnet else None)
            # Check the OUTPUT layer, not any parameter. Both reductions
            # zero-init only their last layer -- the free reduction's hidden
            # Linear(1280,128) is randomly initialised, so "any parameter is
            # non-zero" reports an untrained path as trained.
            proj = getattr(pc, "project", None) if pc is not None else None
            out_w = None
            if proj is not None:
                for mod in reversed(list(proj)):
                    if hasattr(mod, "weight"):
                        out_w = mod.weight
                        break
            trained = out_w is not None and float(out_w.detach().abs().sum()) > 0
            why = ("the pocket path IS trained, so this is the measured weakness "
                   "of pocket conditioning on this input rather than a dead "
                   "path -- see docs/step3_pocket_capacity_2026-09-09.md"
                   if trained else
                   "this checkpoint's pocket reduction is UNTRAINED (zero-init), "
                   "so the pocket contributes exactly zero")
            warnings.warn(
                f"POCKET HAD NO EFFECT: the ranking is identical with and "
                f"without it. Here {why}. Treat this result as "
                f"flavour-conditioned only.",
                RuntimeWarning, stacklevel=2)

    reference = molecule_scores(smiles)

    # --- AutoDock Vina against the receptor that supplied the pocket
    vina_map = None
    redock = None
    pose_files = None
    receptor_atoms = None
    receptor_pdb = None
    if dock:
        if receptor is None:
            raise ValueError(
                "dock=True needs `receptor`: the structure to dock into. Use "
                "the SAME structure the pocket embedding came from, or the "
                "score is measuring a different site than the model was "
                "conditioned on."
            )
        from docking import DockingUnavailable, VinaDocker

        try:
            docker = VinaDocker.from_structure(
                receptor, ligand_resname=ligand_resname,
                exhaustiveness=exhaustiveness)
            # Validate the setup BEFORE trusting any number out of it. A wrong
            # protonation or a mis-centred box returns plausible energies.
            # The control redocks the CRYSTAL ligand, not the input compound --
            # only the crystal ligand has a known pose to compare against.
            redock = docker.redock_control(ligand_resname_smiles or smiles)
            products = [s.product for sl in results for s in sl.suggestions
                        if s.product]
            poses = Path(pose_dir) if pose_dir else None
            targets = sorted(set(products + [smiles]))
            vina_map = docker.dock_many(targets, pose_dir=poses)
            if poses:
                pose_files = {s: poses / f"pose_{i:03d}.sdf"
                              for i, s in enumerate(targets)}
            receptor_atoms = getattr(docker, "_pocket_atoms", None)
            receptor_pdb = getattr(docker, "receptor_pdb", None)
            reference["vina_score"] = vina_map.get(smiles)
        except DockingUnavailable as exc:
            import warnings

            warnings.warn(f"docking skipped: {exc}", RuntimeWarning, stacklevel=2)
            vina_map = None

    table, failures = build_tables(results, smiles, reference, vina_map)
    gallery_path = None
    if gallery:
        note = ""
        if changed is False:
            note = "POCKET SUPPLIED BUT INERT -- ranking identical without it"
        gallery_path = render_gallery(results, smiles, Path(gallery),
                                      flavor_condition=labels, note=note,
                                      reference=reference, table=table,
                                      poses=pose_files,
                                      receptor_atoms=receptor_atoms,
                                      receptor_pdb=receptor_pdb)

    return LeadOptimizationReport(
        input_smiles=smiles, flavor_condition=labels, reference=reference,
        table=table, failures=failures,
        compounds=[p for p in table["product"].tolist()] if len(table) else [],
        gallery=gallery_path, results=results,
        pocket_used=pocket_condition is not None,
        pocket_changed_ranking=changed,
        redock_rmsd=redock,
        docked_receptor=str(receptor) if (dock and vina_map is not None) else None,
        pose_dir=str(pose_dir) if (pose_dir and vina_map is not None) else None)
