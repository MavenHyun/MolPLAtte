#!/usr/bin/env python3
"""Resolve the T1 cognate ligands of the tastepocket set into validated molecules.

Reads ``taste_odor_pdb.json`` and NOT the sibling TSV: one ligand name in the
TSV contains a literal newline, which splits an entry across rows 271-272 and
yields a phantom row whose ``tier`` field reads ``PT5``. The JSON is intact and
its tier counts (343/278/228) sum to its 849 entries.

Bond orders come from ``structures/ligands/<CODE>_ideal.sdf`` -- the CCD ideal
coordinates carry deposited bond orders. Inferring them from the complex
geometry instead is unreliable for exactly the charged groups that matter here
(propionate in 8F76 comes back as ``CCC(O)O``, not ``CCC(=O)[O-]``).

Writes one JSONL record per distinct CCD code.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdkit import Chem, RDLogger  # noqa: E402

from molplatte_prep.pocket_ligands import assess_ligand  # noqa: E402

RDLogger.DisableLog("rdApp.*")

TIER = "T1_ligand_complex"


def load_t1(root: Path) -> list:
    entries = json.loads((root / "data" / "taste_odor_pdb.json").read_text())
    return [e for e in entries if e.get("tier") == TIER]


def mol_from_ideal(sdf: Path):
    """First molecule of a CCD ideal SDF, or None."""
    if not sdf.exists():
        return None
    supplier = Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=True)
    for mol in supplier:
        if mol is not None:
            return mol
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.home() / "datasets/tastepocket")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-cofactors", action="store_true")
    args = ap.parse_args()

    t1 = load_t1(args.root)

    # A ligand code is shared across entries; keep every complex it appears in
    # so the pocket stage can expand one ligand back into its pairs.
    names: dict = {}
    mws: dict = {}
    entries_of = defaultdict(list)
    for e in t1:
        for lig in e.get("cognate_ligands") or []:
            code, name, mw = (lig + [None, None])[:3]
            names.setdefault(code, name)
            mws.setdefault(code, mw)
            entries_of[code].append(
                {
                    "pdb_id": e["id"],
                    "families": e.get("families") or [],
                    "cats": e.get("cats") or [],
                    "uniprot": e.get("uniprot") or [],
                    "organism": e.get("sensor_organism") or "",
                    "method": e.get("method") or "",
                    "resolution": e.get("resolution"),
                }
            )

    ligdir = args.root / "structures" / "ligands"
    reasons: Counter = Counter()
    no_sdf = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with args.out.open("w") as fh:
        for code in sorted(entries_of):
            sdf = ligdir / f"{code}_ideal.sdf"
            mol = mol_from_ideal(sdf)
            if mol is None and sdf.exists():
                reasons["sdf_unparsable"] += 1
            if not sdf.exists():
                no_sdf.append(code)

            verdict = assess_ligand(
                mol,
                code,
                keep_cofactors=args.keep_cofactors,
                # tastepocket is curated as chemosensory complexes, so the
                # CrossDocked-calibrated buffer list is wrong for it.
                allow_chemosensory=True,
            )
            reasons["KEPT" if verdict.ok else verdict.reason] += 1
            if verdict.ok:
                kept += 1

            inchikey = ""
            if mol is not None:
                try:
                    inchikey = Chem.MolToInchiKey(mol)
                except Exception:
                    inchikey = ""

            fh.write(
                json.dumps(
                    {
                        "ccd": code,
                        "name": names.get(code),
                        "ccd_mw": mws.get(code),
                        "ok": verdict.ok,
                        "reason": verdict.reason,
                        "smiles": verdict.smiles,
                        "inchikey": inchikey,
                        "heavy_atoms": verdict.heavy_atoms,
                        "mw": round(verdict.mw, 3),
                        "has_ideal_sdf": sdf.exists(),
                        "n_complexes": len(entries_of[code]),
                        "complexes": entries_of[code],
                    }
                )
                + "\n"
            )

    n_codes = len(entries_of)
    print(f"T1 entries          {len(t1)}")
    print(f"distinct CCD codes  {n_codes}")
    print(f"ideal SDF missing   {len(no_sdf)}  {no_sdf[:12]}")
    print(f"kept                {kept}  ({100 * kept / n_codes:.1f}%)")
    print("\nverdicts:")
    for reason, n in reasons.most_common():
        print(f"  {reason:24s} {n:4d}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
