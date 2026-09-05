#!/usr/bin/env python3
"""Aggregate pocket sites to (ligand, uniprot) records and assign CV folds.

GRAIN. One record per (ligand, receptor), not per pocket instance. Tryptophan
at four genuinely different receptors is four datapoints; 160 crystallographic
copies of alanine in one structure is one. The instance grain inflates the
R-group frequency prior unevenly, and the prior is both what every Hit@K is
judged against and what the logQ correction subtracts -- so inflating it does
not just add duplicates, it moves the number the model is optimising.

FIVE FOLDS, NOT ONE SPLIT. 269 records is too few for a single holdout: a 15%
test set is ~40 records, and the seed would move the answer more than the model
does. Five folds give five estimates and spend every record. The deliverable
checkpoint is then trained on all 269 with no holdout at all -- folds exist to
size the effect, not to produce the artifact.

FOLDS CUT WHOLE COMPONENTS. A fold must be disjoint from the rest on BOTH
sides: a held-out receptor is worthless if its ligand was trained on elsewhere,
because the model has then seen the exact R-groups it is asked to retrieve.
Ligands and receptors are the two node sets of a bipartite graph and a record is
an edge, so cutting whole CONNECTED COMPONENTS makes both disjointness
properties hold by construction and drops nothing. Splitting one side and
patching the other cost 64 of 269 records and still left the halves unbalanced.
The graph cooperates: 45 components, largest holding 20.4% of records.

Pooled ESM-2 pocket embeddings identify the receptor family with 98.5% 1-NN
accuracy across different PDB entries, which is exactly why none of this can be
random: a random split puts the same protein on both sides and reports
memorisation as generalisation.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

#: Evaluation folds. The final checkpoint ignores these and trains on all of it.
N_FOLDS = 5


def uniprot_key(rec: dict) -> str:
    """Receptor identity. Falls back to the PDB entry when unannotated, which
    keeps such records splittable instead of merging unrelated proteins."""
    ups = sorted(rec.get("uniprot") or [])
    return ";".join(ups) if ups else f"noUP:{rec['pdb_id']}"


def connected_components(groups) -> Dict[tuple, int]:
    """(ligand, receptor) -> component id, over the bipartite record graph."""
    parent: Dict[tuple, tuple] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for ccd, up in groups:
        union(("L", ccd), ("R", up))

    roots: Dict[tuple, int] = {}
    out: Dict[tuple, int] = {}
    for ccd, up in groups:
        r = find(("L", ccd))
        out[(ccd, up)] = roots.setdefault(r, len(roots))
    return out


#: Weight of family spread against record-count balance when placing a
#: component. Large enough to pull a family's second component into a different
#: fold; small enough that the folds stay near-equal in size.
FAMILY_SPREAD_WEIGHT = 0.6


def assign_folds(comp_of: Dict[tuple, int], family_of: Dict[int, str],
                 n_folds: int, seed: int) -> Dict[int, int]:
    """Component id -> fold index, balancing record counts AND spreading families.

    Largest-first greedy. Largest-first matters on its own: with components of
    55 and 52 against a fifth of 269, filling smallest-first strands the big
    ones and skews the folds.

    Count balance alone is not enough. Components follow family lines, because
    receptors within a family share ligands -- so packing purely by deficit puts
    a whole family in one fold. The second term prefers a fold that is short of
    the component's family, which spreads the 68% of records whose family spans
    more than one component.

    The other 32% cannot be spread at any weight: TRPV1, CaSR, TRPA1, OTOP1 and
    CSP are each a single component. Those are held out entirely in exactly one
    fold and trained on in the other four, which measures generalisation to an
    unseen receptor FAMILY rather than an unseen receptor. That is a harder
    question than the one the other folds ask, and worth reading separately.
    """
    sizes: Counter = Counter(comp_of.values())
    total = sum(sizes.values())
    per_fold = total / n_folds

    family_total: Counter = Counter()
    for comp, n in sizes.items():
        family_total[family_of.get(comp, "(none)")] += n

    taken: Counter = Counter()
    family_taken: Dict[str, Counter] = defaultdict(Counter)

    rng = random.Random(seed)
    order = sorted(sizes, key=lambda c: (-sizes[c], rng.random()))

    out: Dict[int, int] = {}
    for comp in order:
        fam = family_of.get(comp, "(none)")
        fam_n = max(family_total[fam], 1)

        def score(k: int) -> float:
            deficit = (per_fold - taken[k]) / total
            concentration = family_taken[fam][k] / fam_n
            return deficit - FAMILY_SPREAD_WEIGHT * concentration

        k = max(range(n_folds), key=score)
        out[comp] = k
        taken[k] += sizes[comp]
        family_taken[fam][k] += sizes[comp]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pockets", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--flavor", type=Path, required=True)
    ap.add_argument("--ligands", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=20260905)
    args = ap.parse_args()

    pockets = [json.loads(l) for l in args.pockets.open()]
    lig = {json.loads(l)["ccd"]: json.loads(l) for l in args.ligands.open()}
    flavor = {json.loads(l)["ccd"]: json.loads(l) for l in args.flavor.open()}

    npz = np.load(args.embeddings, allow_pickle=True)
    emb = {str(k): v for k, v in zip(npz["keys"], npz["embeddings"])}
    dim = int(npz["embeddings"].shape[1])

    # -- aggregate to (ligand, uniprot) ------------------------------------
    groups: Dict[tuple, dict] = {}
    missing_emb = 0
    for r in pockets:
        key = (r["ccd"], uniprot_key(r))
        g = groups.setdefault(key, {
            "ccd": r["ccd"], "uniprot": key[1], "families": set(), "cats": set(),
            "organisms": set(), "pdb_ids": set(), "instances": [], "vecs": [],
            "n_pocket_residues": [],
        })
        g["families"].update(r["families"])
        g["cats"].update(r["cats"])
        if r.get("organism"):
            g["organisms"].add(r["organism"])
        g["pdb_ids"].add(r["pdb_id"])
        inst = f"{r['pdb_id']}_{r['ccd']}_{r['instance']}"
        g["instances"].append(inst)
        g["n_pocket_residues"].append(r["n_pocket_residues"])
        v = emb.get(inst)
        if v is None:
            missing_emb += 1
        else:
            g["vecs"].append(v)

    # -- split whole components: both sides disjoint by construction -------
    comp_of = connected_components(groups.keys())
    # A component's family is the one most of its records carry.
    comp_families: Dict[int, Counter] = defaultdict(Counter)
    for (ccd, up), g in groups.items():
        fam = sorted(g["families"])[0] if g["families"] else "(none)"
        comp_families[comp_of[(ccd, up)]][fam] += 1
    family_of = {c: fams.most_common(1)[0][0] for c, fams in comp_families.items()}
    fold_of_comp = assign_folds(comp_of, family_of, args.folds, args.seed)
    n_components = len(set(comp_of.values()))

    records, dropped = [], Counter()
    for (ccd, up), g in sorted(groups.items()):
        fold = fold_of_comp[comp_of[(ccd, up)]]
        if not g["vecs"]:
            dropped["no pocket embedding"] += 1
            continue
        l = lig.get(ccd, {})
        f = flavor.get(ccd, {})
        records.append({
            "id": f"{ccd}__{up}",
            "ccd": ccd,
            "uniprot": up,
            "fold": fold,
            "component": comp_of[(ccd, up)],
            "smiles": l.get("smiles", ""),
            "inchikey": l.get("inchikey", ""),
            "name": l.get("name"),
            "flavor_labels": f.get("labels", []),
            "flavor_source": f.get("source", "unlabelled"),
            "families": sorted(g["families"]),
            "cats": sorted(g["cats"]),
            "organisms": sorted(g["organisms"]),
            "pdb_ids": sorted(g["pdb_ids"]),
            "n_instances": len(g["instances"]),
            "mean_pocket_residues": round(float(np.mean(g["n_pocket_residues"])), 1),
            "pocket_embedding": np.mean(np.stack(g["vecs"]), axis=0).round(5).tolist(),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    # -- report ------------------------------------------------------------
    print(f"pocket sites            {len(pockets)}")
    print(f"(ligand, uniprot) groups{len(groups):>8}")
    print(f"records written         {len(records)}")
    print(f"embedding dim           {dim}")
    print(f"connected components    {n_components}")
    if missing_emb:
        print(f"instances with no embedding: {missing_emb}")
    print("\ndropped:")
    for k, v in dropped.most_common():
        print(f"  {v:4d}  {k}")

    print(f"\nfolds ({args.folds}-fold CV; the final checkpoint uses all records)")
    print(f"  {'fold':>4} {'records':>8} {'ligands':>8} {'receptors':>10}")
    for k in range(args.folds):
        rs = [r for r in records if r["fold"] == k]
        print(f"  {k:>4} {len(rs):>8} {len({r['ccd'] for r in rs}):>8} "
              f"{len({r['uniprot'] for r in rs}):>10}")

    # Every fold must be disjoint from its own training set on BOTH sides.
    bad = 0
    for k in range(args.folds):
        held = [r for r in records if r["fold"] == k]
        rest = [r for r in records if r["fold"] != k]
        lig_ov = {r["ccd"] for r in held} & {r["ccd"] for r in rest}
        rec_ov = {r["uniprot"] for r in held} & {r["uniprot"] for r in rest}
        if lig_ov or rec_ov:
            bad += 1
            print(f"  LEAK fold {k}: {len(lig_ov)} ligands, {len(rec_ov)} receptors")
    print(f"  leakage check: {'OK - every fold disjoint on both sides' if not bad else 'FAILED'}")

    print("\nfamily coverage per fold (a 0 row means that fold tests an unseen family):")
    fams = sorted({r["families"][0] if r["families"] else "(none)" for r in records})
    header = "".join(f"{k:>5}" for k in range(args.folds))
    print(f"  {'family':50s}{header}")
    for fam in fams:
        row = [sum(1 for r in records if r["fold"] == k
                   and (r["families"][0] if r["families"] else "(none)") == fam)
               for k in range(args.folds)]
        flag = "  <- single-component family" if sum(1 for v in row if v) == 1 else ""
        print(f"  {fam[:50]:50s}" + "".join(f"{v:>5}" for v in row) + flag)

    src = Counter(r["flavor_source"] for r in records)
    print("\nflavor source:", dict(src.most_common()))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
