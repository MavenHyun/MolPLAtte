#!/usr/bin/env python3
"""Read the STEP 2 fold logs and report the cross-validated result.

Reports each fold separately BEFORE any average, because the folds do not all
ask the same question. TRPV1, CaSR, TRPA1, OTOP1 and CSP are each a single
connected component of the (ligand, receptor) graph, so they are held out
wholesale in exactly one fold -- that fold measures generalisation to an unseen
receptor FAMILY, which is strictly harder than the unseen-receptor question the
other folds ask. Averaging them together hides that.

The headline number is ``novel_hit@K``, not ``hit@K``. A healthy hit@K with a
collapsed novel_hit@K means the model is retrieving chemistry it already knew,
which the pocket stage was not built to demonstrate.
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional

# [RGroupLibraryRetrieval/val] library=91,935 eff=946 N=20,000 MRR=0.3690
#   H@1=0.2715(prior 0.0244)  H@10=0.5607(prior 0.0913) ...
LINE = re.compile(
    r"RGroupLibraryRetrieval/(?P<stage>\w+)\].*?"
    r"N=(?P<n>[\d,]+)\s+MRR=(?P<mrr>[\d.]+)"
)
HIT = re.compile(r"H@(?P<k>\d+)=(?P<v>[\d.]+)\(prior (?P<prior>[\d.]+)\)")
SPLIT = re.compile(r"(?P<half>base|novel)_hit@(?P<k>\d+)[=:]\s*(?P<v>[\d.]+)")


def last_eval(log: Path) -> Optional[dict]:
    """Final validation line of a fold log, or None if it never got there."""
    best = None
    for line in log.read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        row = {
            "n": int(m.group("n").replace(",", "")),
            "mrr": float(m.group("mrr")),
            "hit": {int(h.group("k")): float(h.group("v")) for h in HIT.finditer(line)},
            "prior": {int(h.group("k")): float(h.group("prior"))
                      for h in HIT.finditer(line)},
        }
        for s in SPLIT.finditer(line):
            row.setdefault(s.group("half"), {})[int(s.group("k"))] = float(s.group("v"))
        best = row
    return best


def failure_reason(log: Path) -> str:
    text = log.read_text(errors="replace")
    for marker in ("Traceback", "CUDA out of memory", "RuntimeError", "Killed"):
        if marker in text:
            return marker
    if "FINISHED ====> Fitting Trainer" in text:
        return "finished but logged no retrieval eval"
    return "still running or died silently"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", type=Path,
                    default=Path.home() / "checkpoints/molplatte/hplogs")
    ap.add_argument("--pattern", default="s2-pocket-fold*.log")
    #: Folds whose held-out set is dominated by a single-component family.
    ap.add_argument("--family-holdout-folds", default="1",
                    help="comma-separated; reported apart from the rest")
    args = ap.parse_args()

    logs = sorted(args.logdir.glob(args.pattern))
    if not logs:
        print(f"no fold logs matching {args.pattern} in {args.logdir}")
        return 1

    family_folds = {int(x) for x in args.family_holdout_folds.split(",") if x.strip()}
    rows: Dict[int, dict] = {}
    for log in logs:
        m = re.search(r"fold(\d+)", log.name)
        if not m:
            continue
        fold = int(m.group(1))
        row = last_eval(log)
        if row is None:
            print(f"  fold {fold}: NO RESULT -- {failure_reason(log)}")
            continue
        rows[fold] = row

    if not rows:
        print("no fold produced a retrieval evaluation")
        return 1

    cuts = sorted(next(iter(rows.values()))["hit"])
    head = "  fold   N      MRR " + "".join(f"   H@{k:<4}" for k in cuts)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for fold in sorted(rows):
        r = rows[fold]
        tag = "  <- unseen family" if fold in family_folds else ""
        cells = "".join(f"  {r['hit'].get(k, float('nan')):.4f}" for k in cuts)
        print(f"  {fold:<5}{r['n']:<7}{r['mrr']:.4f}{cells}{tag}")

    ordinary = [f for f in rows if f not in family_folds]
    if ordinary:
        print(f"\n  unseen-receptor folds ({', '.join(map(str, sorted(ordinary)))}):")
        for k in cuts:
            vals = [rows[f]["hit"][k] for f in ordinary if k in rows[f]["hit"]]
            pri = [rows[f]["prior"][k] for f in ordinary if k in rows[f]["prior"]]
            if not vals:
                continue
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            lift = (statistics.mean(vals) / statistics.mean(pri)
                    if pri and statistics.mean(pri) else float("nan"))
            print(f"    H@{k:<4} {statistics.mean(vals):.4f} +- {sd:.4f}"
                  f"   prior {statistics.mean(pri):.4f}   lift {lift:.1f}x")

    seen_family = [f for f in rows if f in family_folds]
    if seen_family:
        print(f"\n  unseen-FAMILY folds ({', '.join(map(str, sorted(seen_family)))}) "
              "-- harder question, do not average with the above:")
        for f in sorted(seen_family):
            cells = "  ".join(f"H@{k}={rows[f]['hit'][k]:.4f}"
                              for k in cuts if k in rows[f]["hit"])
            print(f"    fold {f}: {cells}")

    # base vs novel, the number that says whether the pocket stage generalised
    have_split = [f for f in rows if "novel" in rows[f]]
    print()
    if not have_split:
        print("  base/novel split NOT in these logs -- the run scored against a "
              "single-source library, or the union provenance was missing. "
              "Without it, hit@K cannot distinguish 'retrieved new chemistry' "
              "from 'retrieved what it already knew'.")
    else:
        print("  base vs novel (novel = absent from the pretraining vocabulary):")
        for k in cuts:
            b = [rows[f]["base"][k] for f in have_split if k in rows[f].get("base", {})]
            n = [rows[f]["novel"][k] for f in have_split if k in rows[f].get("novel", {})]
            if b and n:
                print(f"    H@{k:<4} base {statistics.mean(b):.4f}   "
                      f"novel {statistics.mean(n):.4f}")
        print("\n  A healthy base_hit@K beside a collapsed novel_hit@K means the "
              "model is retrieving chemistry it already knew.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
