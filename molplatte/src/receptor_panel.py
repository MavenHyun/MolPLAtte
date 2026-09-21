"""A curated panel of taste/odour receptor structures to dock generated compounds into.

The pocket-LESS optimiser proposes compounds from flavour alone. This panel adds
a structural opinion afterwards: dock each product into the receptors that
actually mediate the requested percept, and report the binding scores beside the
retrieval score.

It is deliberately NOT model conditioning. Docking is independent of the model
and free to disagree with it -- which is the only reason it is worth computing.

Each entry is validated by REDOCKING ITS OWN CRYSTAL LIGAND before use. A
receptor whose own ligand does not land near its known pose cannot be trusted to
rank novel compounds, and a docking setup fails silently otherwise. Entries that
fail are dropped, loudly.

Human structures are preferred. Where the only structure is non-human the entry
says so, because a score against a fish or mouse receptor is weaker evidence
about a human percept.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

STRUCTURES = Path.home() / "datasets" / "tastepocket" / "structures" / "cif"
LIGANDS = Path.home() / "preprocessed" / "molplatte" / "tastepocket" / "ligands.jsonl"
CACHE = Path.home() / "preprocessed" / "molplatte" / "receptor_panel_validation.json"

#: Flavour terms come from molplatte_prep.condvec.FLAVOR_LABELS.
@dataclass(frozen=True)
class Receptor:
    key: str
    pdb: str
    ccd: str
    family: str
    organism: str
    resolution: float
    #: flavour labels this receptor is evidence for
    flavours: tuple


PANEL: List[Receptor] = [
    Receptor("T2R_bitter", "8VY7", "A1AEI", "T2R bitter taste receptor",
             "Homo sapiens", 2.68, ("bitter",)),
    # 9PEA over the higher-resolution 11CL: 11CL redocks at 2.52 A (over the
    # 2.5 A bar) while 9PEA lands at 0.62 A, and 9PEA's ligand is a vanillylamide
    # -- the capsaicin site itself, which is the pharmacology "spicy" means.
    Receptor("TRPV1_pungent", "9PEA", "A1CHV", "TRPV1 capsaicin/vanilloid",
             "Homo sapiens", 2.60, ("spicy",)),
    # 9MOE over the higher-resolution 9VU1: 9VU1's ligand is liquiritin
    # apioside, a large flexible glycoside that redocks at 5.93 A median. 9MOE
    # lands at 0.36 A with a 0.02 A spread across seeds. Resolution does not
    # predict dockability; the control does.
    Receptor("TRPA1_irritant", "9MOE", "A1BNO", "TRPA1 irritant/pungency",
             "Homo sapiens", 2.70, ("spicy", "sulfurous")),
    # 7XJ1 over 7XJ0: SAME ligand, but 7XJ0 redocks at 7.43 A against 7XJ1's
    # 1.07 A. 7XJ0 is a TRPV3/3C-GFP fusion construct -- the failure is the
    # structure, not the chemistry, which is exactly what a redock control is for.
    Receptor("TRPV3_warm", "7XJ1", "EQK", "TRPV3 warmth/camphor",
             "Homo sapiens", 2.93, ("herbal", "woody", "medicinal")),
    Receptor("TRPM8_cooling", "9PB5", "XUQ", "TRPM8 cooling/menthol",
             "Homo sapiens", 3.50, ("minty",)),
    Receptor("CaSR_kokumi", "5FBK", "TCR", "CaSR kokumi/calcium-sensing",
             "Homo sapiens", 2.10, ("umami", "meaty")),
    Receptor("T1R_sweet_umami", "9OQ6", "A1CD7", "T1R sweet/umami",
             "Homo sapiens", 3.57, ("sweet", "umami")),
    Receptor("PKD2L1_sour", "8HK7", "AQV", "OTOP1/PKD2L1 sour channel",
             "Homo sapiens", 3.00, ("sour",)),
    Receptor("OR_olfactory", "8HTI", "OCA", "Olfactory receptor (vertebrate GPCR)",
             "Homo sapiens", 2.97, ("fruity", "green", "floral", "citrus",
                                    "earthy", "nutty", "roasted", "dairy",
                                    "alcoholic", "fatty")),
]

#: redock RMSD above this means the setup does not reproduce a known answer
REDOCK_MAX_A = 2.5


def _ligand_smiles() -> Dict[str, str]:
    out = {}
    for line in open(LIGANDS):
        r = json.loads(line)
        if r.get("smiles"):
            out[r["ccd"]] = r["smiles"]
    return out


def receptors_for(flavours: Sequence[str]) -> List[Receptor]:
    """Panel entries relevant to the requested percept, most specific first."""
    want = {f.lower().strip() for f in flavours}
    hits = [r for r in PANEL if want & set(r.flavours)]
    return sorted(hits, key=lambda r: (len(r.flavours), r.resolution))


def validate(keys: Optional[Sequence[str]] = None, *, refresh: bool = False,
             exhaustiveness: int = 8) -> Dict[str, dict]:
    """Redock each panel member's own crystal ligand. Cached to CACHE."""
    from docking import VinaDocker  # flat module layout, matching lead_report.py

    cached = {}
    if CACHE.exists() and not refresh:
        cached = json.loads(CACHE.read_text())
    smi = _ligand_smiles()
    todo = [r for r in PANEL if (keys is None or r.key in keys) and r.key not in cached]
    for r in todo:
        path = STRUCTURES / f"{r.pdb}.cif"
        rec = {"pdb": r.pdb, "ccd": r.ccd, "organism": r.organism,
               "resolution": r.resolution}
        try:
            # THREE seeds, median. A single redock is a coin flip: one entry
            # scored 1.45 A and then 5.38 A on identical inputs before the Vina
            # seed was pinned. Requiring the MEDIAN to pass tests whether the
            # setup is reliable, not whether one sample got lucky.
            rmsds = []
            for sd in (0xC0FFEE, 0x5EED, 0xBEEF):
                d = VinaDocker.from_structure(path, ligand_resname=r.ccd,
                                              exhaustiveness=exhaustiveness,
                                              seed=sd)
                val = d.redock_control(smi.get(r.ccd, ""))
                if val is not None:
                    rmsds.append(float(val))
            rmsd = (sorted(rmsds)[len(rmsds) // 2] if rmsds else None)
            rec["redock_rmsd"] = rmsd
            rec["redock_all"] = rmsds
            rec["ok"] = bool(rmsd is not None and rmsd <= REDOCK_MAX_A)
        except Exception as exc:  # noqa: BLE001
            rec["redock_rmsd"] = None
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
        cached[r.key] = rec
        logger.info("panel %s: %s", r.key, rec)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cached, indent=1))
    return cached


def usable(validation: Optional[Dict[str, dict]] = None) -> List[Receptor]:
    v = validation if validation is not None else validate()
    return [r for r in PANEL if v.get(r.key, {}).get("ok")]


def dock_panel(smiles: Sequence[str], flavours: Sequence[str], *,
               all_receptors: bool = False, exhaustiveness: int = 8,
               pose_dir: Optional[Path] = None,
               progress: bool = True) -> Dict[str, Dict[str, Optional[float]]]:
    """Dock every compound into the flavour-relevant panel receptors.

    Returns ``{receptor_key: {smiles: affinity_kcal_per_mol}}``. A compound that
    fails to dock is ``None`` rather than absent, so a missing score is visible
    as a gap instead of vanishing from the table.

    Only VALIDATED receptors are used -- one whose own crystal ligand does not
    redock is excluded, because it cannot be trusted to rank anything else.
    """
    from docking import VinaDocker

    v = validate()
    ok = {r.key for r in usable(v)}
    picks = PANEL if all_receptors else receptors_for(flavours)
    picks = [r for r in picks if r.key in ok]
    if not picks:
        logger.warning("no validated panel receptor matches %s", list(flavours))
        return {}

    out: Dict[str, Dict[str, Optional[float]]] = {}
    ref: Dict[str, Optional[float]] = {}
    smi = _ligand_smiles()
    for r in picks:
        if progress:
            print(f"  docking {len(smiles)} compounds into {r.key} "
                  f"({r.pdb}, {r.organism}, {r.resolution:.2f} A)", flush=True)
        try:
            d = VinaDocker.from_structure(STRUCTURES / f"{r.pdb}.cif",
                                          ligand_resname=r.ccd,
                                          exhaustiveness=exhaustiveness)
        except Exception as exc:  # noqa: BLE001
            logger.warning("panel %s unavailable: %s", r.key, exc)
            continue
        # The receptor's OWN crystal ligand, as a per-receptor reference.
        # Vina scores are NOT comparable across pockets: a more enclosed site
        # scores more negative regardless of fit, so the raw minimum across
        # receptors just names the tightest pocket. Measured here: TRPM8 beat
        # PKD2L1 on 40/40 compounds with a near-constant -0.73 kcal/mol offset.
        # Referencing each compound to the ligand crystallised in that same
        # pocket cancels the offset and makes "better than native" meaningful.
        native = None
        ref_smiles = smi.get(r.ccd)
        if ref_smiles:
            try:
                native = d.dock(ref_smiles)
            except Exception:  # noqa: BLE001
                native = None
        ref[r.key] = native
        sub = pose_dir / r.key if pose_dir else None
        scores = {}
        for s in smiles:
            try:
                scores[s] = d.dock(s, pose_out=(sub / f"{abs(hash(s))%10**9}.sdf")
                                   if sub else None)
            except Exception:  # noqa: BLE001
                scores[s] = None
        out[r.key] = scores
    out['__native__'] = ref
    return out


def panel_columns(df, smiles_col: str, flavours: Sequence[str], **kw):
    """Attach one ``vina_<receptor>`` column per relevant receptor, plus a summary.

    ``panel_best`` is the strongest (most negative) affinity across the panel and
    ``panel_best_receptor`` names where it came from -- the two columns a reader
    actually sorts on.
    """
    import pandas as pd  # noqa: WPS433

    smiles = [s for s in df[smiles_col].tolist() if isinstance(s, str)]
    res = dock_panel(smiles, flavours, **kw)
    if not res:
        return df
    native = res.pop("__native__", {})
    out = df.copy()
    dcols = []
    for key, scores in res.items():
        out[f"vina_{key}"] = out[smiles_col].map(scores)
        n = native.get(key)
        if n is not None:
            # negative = binds better than the ligand crystallised in that pocket
            out[f"dvina_{key}"] = out[f"vina_{key}"] - float(n)
            dcols.append(f"dvina_{key}")
    if dcols:
        sub = out[dcols]
        out["panel_best"] = sub.min(axis=1)
        out["panel_best_receptor"] = sub.idxmin(axis=1).str.replace("dvina_", "", regex=False)
    else:
        cols = [f"vina_{k}" for k in res]
        sub = out[cols]
        out["panel_best"] = sub.min(axis=1)
        out["panel_best_receptor"] = sub.idxmin(axis=1).str.replace("vina_", "", regex=False)
    return out
