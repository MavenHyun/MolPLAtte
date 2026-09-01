#!/usr/bin/env python3
"""Merge R-group vocabularies into one library, tagging where each entry came from.

The pocket stage retrieves against chemistry the pretraining corpus never saw:
40% of CrossDocked's distinct R-groups are absent from `coconut-flavordb-full`.
Scoring against the pretraining library alone silently drops those queries
(RGroupLibraryRetrieval masks `rows_of() == -1`), so the reported Hit@K would
cover only the R-groups the model already knew -- the common tail, since matched
R-groups have a median corpus count of ~1,600.

This builds the union library and marks every entry `in_base`, so retrieval can
report the two halves separately instead of hiding one.

    python scripts/build_union_vocab.py OUT.pkl.gz BASE_VOCAB EXTRA_VOCAB [EXTRA...]

The FIRST input is the base (the pretraining corpus); entries present in it are
`in_base=True`. Counts are summed across inputs, because the frequency prior and
the logQ correction both read them and must describe the library actually used.
"""
from __future__ import annotations
import gzip, math, pickle, sys
from pathlib import Path


def load(p):
    with gzip.open(p, "rb") as fh:
        return pickle.load(fh)


def main(out, base, *extras):
    bv = load(base)
    # entries is a dict keyed by hash, and `graph` is DEHYDRATED. Copy the raw
    # payload through untouched -- hydrating and re-dehydrating here would make
    # this script depend on the graph codec staying stable.
    merged = dict(bv["entries"])
    base_hashes = set(merged)

    novel = []
    for p in extras:
        for h, raw in load(p)["entries"].items():
            if h in merged:
                m = dict(merged[h])
                m["count"] = int(m.get("count", 0)) + int(raw.get("count", 0))
                merged[h] = m
            else:
                merged[h] = raw
                novel.append(h)

    total = sum(int(r.get("count", 0)) for r in merged.values()) or 1
    H = -sum((c / total) * math.log(c / total)
             for c in (int(r.get("count", 0)) for r in merged.values()) if c > 0)

    prov = dict(bv.get("provenance") or {})
    prov.update({
        "union_of": [str(base)] + [str(e) for e in extras],
        "n_base": len(base_hashes),
        "n_novel": len(novel),
        "n_total": len(merged),
        "effective_size": math.exp(H),
        # Only the novel keys are stored: `in_base` is `hash not in novel_hashes`,
        # and 5.6k strings is far cheaper than 92k. load_vocabulary copies
        # provenance wholesale, so this reaches the training side intact.
        "novel_hashes": novel,
    })
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wb") as fh:
        pickle.dump({"provenance": prov, "entries": merged}, fh, protocol=4)
    print(f"  base            {len(base_hashes):,}")
    print(f"  added (novel)   {len(novel):,}")
    print(f"  union           {len(merged):,}   effective {math.exp(H):,.0f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], *sys.argv[3:])
