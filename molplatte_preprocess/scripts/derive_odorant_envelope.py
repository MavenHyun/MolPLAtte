#!/usr/bin/env python3
"""Derive the odorant physicochemical envelope and measure what filtering COCONUT by it buys.

The `coconut-flavordb-filtered` corpus keeps a COCONUT compound only if it falls
inside an envelope derived from FlavorDB's OWN non-imputed flavor labels -- never
from a hand-picked threshold. This script recomputes that envelope from source and
reports the enrichment, so the corpus is reproducible rather than a remembered number.

Writes JSON to stdout and (optionally) an allowlist of COCONUT identifiers.
"""
from __future__ import annotations
import csv, io, json, sys, zipfile

# COCONUT rows carry huge molblock/descriptor fields
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
from pathlib import Path

FLAVORDB = Path("/home/mogan/datasets/flavordb")
COCONUT_CSV = Path("/home/mogan/datasets/coconut/coconut_csv-08-2026.zip")
MEASURED = Path("/home/mogan/preprocessed/molplatte/annotations/flavor_measured.jsonl")

#: Percentile band taken from real odorants. 5-95 keeps the bulk of measured
#: flavor chemistry while cutting the tails that COCONUT is dense in.
LO, HI = 5.0, 95.0


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * q / 100.0
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def load_flavordb():
    """Real odorants = FlavorDB entries whose profile maps onto a sensory class.

    NOT every row with a non-empty ``flavor_profile``: that column is populated
    for 25,106 of 25,596 rows, including propagated/imputed entries, and using it
    yields a median MW of 346 -- i.e. it describes FlavorDB's bulk, not odorants.
    ``flavor_measured.jsonl`` is the curated subset (11,262 InChIKeys) that
    survived mapping onto the 24 FLAVOR_LABELS, and has median MW ~170.
    """
    measured = set()
    with open(MEASURED) as fh:
        for line in fh:
            line = line.strip()
            if line:
                ik = (json.loads(line).get("inchikey") or "").strip()
                if ik:
                    measured.add(ik)
    odorants, keys = [], set()
    with open(FLAVORDB / "properties.csv") as fh:
        for r in csv.DictReader(fh):
            ik = (r.get("InChIKey") or "").strip()
            if ik:
                keys.add(ik)
            try:
                mw, lp, tp = float(r["MolecularWeight"]), float(r["XLogP"]), float(r["TPSA"])
            except (KeyError, ValueError, TypeError):
                continue
            if ik in measured:
                odorants.append((mw, lp, tp))
    return odorants, measured, len(measured)


def envelope(odorants):
    mw = [o[0] for o in odorants]; lp = [o[1] for o in odorants]; tp = [o[2] for o in odorants]
    return {
        "n_odorants": len(odorants),
        "mw":   [round(pct(mw, LO), 1), round(pct(mw, HI), 1)],
        "logp": [round(pct(lp, LO), 2), round(pct(lp, HI), 2)],
        "tpsa": round(pct(tp, HI), 1),
        "mw_median": round(pct(mw, 50), 1),
    }


def scan_coconut(env, fdb_keys, allowlist_out=None):
    lo_mw, hi_mw = env["mw"]; lo_lp, hi_lp = env["logp"]; hi_tp = env["tpsa"]
    seen = set(); n = n_in = n_known = n_known_in = 0
    keep = []
    with zipfile.ZipFile(COCONUT_CSV) as z:
        with z.open(z.namelist()[0]) as fh:
            for r in csv.DictReader(io.TextIOWrapper(fh, "utf-8")):
                ident = (r.get("identifier") or "").split(".")[0]
                if not ident or ident in seen:
                    continue
                seen.add(ident)
                n += 1
                try:
                    mw, lp, tp = (float(r["molecular_weight"]), float(r["alogp"]),
                                  float(r["topological_polar_surface_area"]))
                except (KeyError, ValueError, TypeError):
                    continue
                known = (r.get("standard_inchi_key") or "").strip() in fdb_keys
                inside = (lo_mw <= mw <= hi_mw and lo_lp <= lp <= hi_lp and tp <= hi_tp)
                n_known += known
                if inside:
                    n_in += 1; n_known_in += known
                # the corpus keeps the envelope UNION every known-flavor compound
                if inside or known:
                    keep.append(ident)
    if allowlist_out:
        Path(allowlist_out).write_text("\n".join(keep) + "\n")
    base = n_known / n if n else 0.0
    dens = n_known_in / n_in if n_in else 0.0
    return {
        "coconut_distinct": n, "inside_envelope": n_in,
        "kept_union_known": len(keep),
        "known_flavor_total": n_known, "known_flavor_inside": n_known_in,
        "base_rate_pct": round(100 * base, 3),
        "inside_density_pct": round(100 * dens, 2),
        "enrichment_x": round(dens / base, 1) if base else None,
        "recall_of_known_pct": round(100 * n_known_in / n_known, 1) if n_known else None,
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    odorants, fdb_keys, n_lab = load_flavordb()
    env = envelope(odorants)
    res = scan_coconut(env, fdb_keys, out)
    print(json.dumps({"flavordb_labelled": n_lab, "flavordb_inchikeys": len(fdb_keys),
                      "envelope": env, "coconut": res}, indent=2))
