#!/usr/bin/env python3
"""What did retrieval actually return? Resolve prediction tables to chemistry.

The prediction tables store WL hashes, which are unreadable. This joins them to
the retrieval library and reports what was asked for and what came back.

The number that matters most is rank-1 DIVERSITY. A model that answers every
query with the same handful of common R-groups can still post a respectable
Hit@K, because those R-groups genuinely are the answer often enough -- the
frequency prior alone scores 0.0249 at K=1 on this library. Collapse of that
kind is invisible in Hit@K and obvious here.

BUT DIVERSITY IS NOT COMPARABLE ACROSS RUNS UNTIL IT IS NORMALISED. PredictionTable
builds its gallery from the batch's own distinct target chemistries, not from the
91,935-row library, so a run over 345 chemistries and a run over 51 are choosing
from very different candidate sets. Raw "distinct returned" makes the smaller run
look collapsed automatically. The reported ratio divides by what uniform selection
would give, G*(1-(1-1/G)^N), which removes the gallery-size effect.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import pickle
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

DEFAULT_VOCAB = (Path.home() / "preprocessed/molplatte/union_vocab"
                 / "base-full__crossdocked__tastepocket" / "rgroup_vocab.pkl.gz")


def load_vocab(path: Path):
    with gzip.open(path, "rb") as fh:
        v = pickle.load(fh)
    entries = v["entries"] if isinstance(v, dict) else v.entries
    prov = v["provenance"] if isinstance(v, dict) else v.provenance
    smiles, counts = {}, {}
    for h, e in entries.items():
        smiles[h] = e["smiles"] if isinstance(e, dict) else e.smiles
        counts[h] = e["count"] if isinstance(e, dict) else e.count
    return smiles, counts, set(prov.get("novel_hashes") or [])


def heavy_atoms(smi: str) -> int:
    if not smi:
        return 0
    m = Chem.MolFromSmiles(smi.replace("*", "[*]"))
    return m.GetNumHeavyAtoms() - smi.count("*") if m else 0


def latest_table(run_dir: Path):
    tables = sorted((run_dir / "prediction_tables").glob("*.csv"))
    if not tables:
        return None
    # Filenames carry the epoch, so the highest sorts to the trained model.
    return max(tables, key=lambda p: p.name)


def analyse(path: Path, smiles, counts, novel, top: int, label: str) -> None:
    rank1, targets, rank1_wrong = Counter(), Counter(), Counter()
    hits = misses = n = 0
    with path.open() as fh:
        for row in csv.DictReader(fh):
            n += 1
            tgt = row["target_hash"]
            ret = [h.strip() for h in row["retrieved_target_hashes"].split("|")
                   if h.strip()]
            if not ret:
                continue
            targets[tgt] += 1
            rank1[ret[0]] += 1
            if ret[0] == tgt:
                hits += 1
            else:
                misses += 1
                rank1_wrong[ret[0]] += 1

    if not n:
        print(f"  {label}: empty table")
        return

    print(f"\n{'='*76}\n{label}   ({path.name})\n{'='*76}")
    print(f"  queries {n}   rank-1 correct {hits} ({100*hits/n:.1f}%)")
    gallery = len(targets)
    expected = gallery * (1 - (1 - 1 / gallery) ** n) if gallery else 0
    print(f"  gallery (distinct chemistries): {gallery}")
    print(f"  distinct R-groups returned @1 : {len(rank1)}"
          f"   vs {expected:.1f} expected under uniform choice"
          f"   ratio {len(rank1)/expected:.2f}" if expected else "")
    if rank1:
        share = rank1.most_common(1)[0][1] / n
        print(f"  single most-returned R-group  : {100*share:.1f}% of queries"
              f"   ({share*gallery:.1f}x the uniform share)")

    def show(counter, title, k):
        print(f"\n  {title}")
        print(f"    {'n':>5} {'share':>7} {'heavy':>6} {'corpus':>10}  smiles")
        for h, c in counter.most_common(k):
            smi = smiles.get(h, "?")
            tag = "  novel" if h in novel else ""
            print(f"    {c:>5} {100*c/n:6.1f}% {heavy_atoms(smi):6d} "
                  f"{counts.get(h,0):10,}  {smi[:32]}{tag}")

    show(rank1, f"returned at rank 1 (top {top})", top)
    show(targets, f"asked for (top {top})", top)

    def med(vals):
        vals = sorted(vals)
        return vals[len(vals) // 2] if vals else 0

    ret_c = [counts.get(h, 0) for h, c in rank1.items() for _ in range(c)]
    tgt_c = [counts.get(h, 0) for h, c in targets.items() for _ in range(c)]
    print(f"\n  corpus count  returned median {med(ret_c):>9,}"
          f"   asked-for median {med(tgt_c):>9,}")
    ret_h = [heavy_atoms(smiles.get(h, "")) for h, c in rank1.items() for _ in range(c)]
    tgt_h = [heavy_atoms(smiles.get(h, "")) for h, c in targets.items() for _ in range(c)]
    print(f"  heavy atoms   returned median {med(ret_h):>9}"
          f"   asked-for median {med(tgt_h):>9}")

    if rank1_wrong:
        print(f"\n  when rank 1 was WRONG ({misses} queries) it returned:")
        for h, c in rank1_wrong.most_common(5):
            print(f"    {c:>5} {100*c/max(misses,1):5.1f}%  {smiles.get(h,'?')[:34]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    smiles, counts, novel = load_vocab(args.vocab)
    print(f"library {len(smiles):,} rows, {len(novel):,} novel")
    for d in args.run_dirs:
        t = latest_table(d)
        if t is None:
            print(f"\n  {d.name}: no prediction tables")
            continue
        analyse(t, smiles, counts, novel, args.top, d.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
