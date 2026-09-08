#!/usr/bin/env python3
"""Is ZINC2020 chemically relevant to flavour compounds, and how much of it?

Run BEFORE building a ZINC corpus. 528M molecules cannot all be preprocessed
(~3.3 TB), so the subsample has to be chosen, and choosing it blindly risks
pretraining on chemistry the flavour stages never see.

Three measurements, cheapest first:

1. **Odorant envelope.** The same MW / logP / TPSA envelope the
   `coconut-flavordb-filtered` corpus is built on, derived in
   `derive_odorant_envelope.py` from FlavorDB's own non-imputed labels rather
   than from a hand-picked threshold. What share of ZINC falls inside it?

2. **R-group overlap.** The model retrieves R-GROUPS, so the question that
   actually matters is not whether whole molecules resemble odorants but
   whether ZINC's R-group vocabulary overlaps the flavour library. Decomposes a
   sample with the corpus settings and joins on WL hash.

3. **Coverage of the flavour library's mass.** Overlap counted over distinct
   R-groups understates the useful part, because the flavour library is
   extremely skewed. Weighting the overlap by corpus occurrence says what
   fraction of the retrieval target space ZINC actually reaches.

Sampling is shard-stratified: ZINC organises shards by molecular weight and
logP tranche, so taking whole shards would bias the estimate toward whichever
tranches were drawn.
"""
from __future__ import annotations

import argparse
import gzip
import random
import sys
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors  # noqa: E402

RDLogger.DisableLog("rdApp.*")

#: Derived in derive_odorant_envelope.py from FlavorDB's non-imputed odorants.
ENV_MW = (108.0, 290.0)
ENV_LOGP = (0.4, 4.5)
ENV_TPSA_MAX = 53.0

DECOMP = dict(method="naveja_recap", ratio=1.0 / 3.0, include_ring=True,
              max_cores=4, min_rgroup_atoms=2)


def in_envelope(mol) -> bool:
    mw = Descriptors.MolWt(mol)
    if not (ENV_MW[0] <= mw <= ENV_MW[1]):
        return False
    lp = Crippen.MolLogP(mol)
    if not (ENV_LOGP[0] <= lp <= ENV_LOGP[1]):
        return False
    return rdMolDescriptors.CalcTPSA(mol) <= ENV_TPSA_MAX


def sample_molecules(root: Path, n_shards: int, per_shard: int, seed: int):
    """Shard-stratified draw. Yields (mol, shard) pairs."""
    shards = [s for s in root.rglob("*.sdf.gz") if s.stat().st_size > 0]
    rng = random.Random(seed)
    rng.shuffle(shards)
    for sh in shards[:n_shards]:
        taken = 0
        try:
            with gzip.open(sh, "rb") as fh:
                for mol in Chem.ForwardSDMolSupplier(fh, removeHs=True, sanitize=True):
                    if mol is None:
                        continue
                    yield mol, sh
                    taken += 1
                    if taken >= per_shard:
                        break
        except Exception:  # noqa: BLE001 - a corrupt shard must not end the scan
            continue


def flavour_library(path: Path):
    """hash -> corpus count for the flavour retrieval library."""
    import gzip as gz
    import pickle

    with gz.open(path, "rb") as fh:
        v = pickle.load(fh)
    e = v["entries"] if isinstance(v, dict) else v.entries
    return {h: (x["count"] if isinstance(x, dict) else x.count) for h, x in e.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zinc", type=Path, default=Path.home() / "datasets/zinc2020/raw")
    ap.add_argument("--library", type=Path,
                    default=Path.home() / "preprocessed/molplatte/coconut-flavordb-full"
                    / "naveja_recap/rgroup_vocab.pkl.gz")
    ap.add_argument("--shards", type=int, default=400)
    ap.add_argument("--per-shard", type=int, default=60)
    ap.add_argument("--decompose", type=int, default=4000,
                    help="how many of the sampled molecules to decompose")
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    print(f"sampling {args.shards} shards x {args.per_shard} molecules ...")
    mols = [m for m, _ in sample_molecules(args.zinc, args.shards,
                                           args.per_shard, args.seed)]
    n = len(mols)
    print(f"  sampled {n:,} molecules\n")

    # -- 1. envelope ------------------------------------------------------
    inside = [m for m in mols if in_envelope(m)]
    print("1. ODORANT ENVELOPE "
          f"(MW {ENV_MW[0]:.0f}-{ENV_MW[1]:.0f}, logP {ENV_LOGP[0]}-{ENV_LOGP[1]}, "
          f"TPSA <= {ENV_TPSA_MAX:.0f})")
    print(f"   inside: {len(inside):,}/{n:,} = {100*len(inside)/n:.1f}%")
    mws = sorted(Descriptors.MolWt(m) for m in mols)
    print(f"   ZINC MW: median {mws[len(mws)//2]:.0f}  "
          f"p5 {mws[int(.05*len(mws))]:.0f}  p95 {mws[int(.95*len(mws))]:.0f}")
    print(f"   => projected usable pool from 528M: "
          f"{528 * len(inside) / n:.0f}M molecules\n")

    # -- 2 & 3. R-group overlap -------------------------------------------
    if not args.library.is_file():
        print(f"   (no library at {args.library}; skipping R-group overlap)")
        return 0
    lib = flavour_library(args.library)
    lib_total = sum(lib.values())
    print(f"2. R-GROUP OVERLAP against {len(lib):,} flavour R-groups")

    from molplatte_prep.decompose import decompose_molecule
    from molplatte_prep.graph_hash import subgraph_hash
    from molplatte_prep.graph_ops import detach_rgroups_multi
    from molplatte_prep.mol_features import mol_to_pyg

    def hashes_of(mol):
        out = []
        try:
            data = mol_to_pyg(mol)
            _, decs = decompose_molecule(mol, do_wash=False, **DECOMP)
        except Exception:  # noqa: BLE001
            return out
        for dec in decs:
            for rg in dec.rgroups:
                try:
                    _t, det = detach_rgroups_multi(
                        data, [(rg.rgroup_atoms, rg.core_linker, rg.rgroup_linker)],
                        store_orig=False)
                    out.append(subgraph_hash(det[0]))
                except Exception:  # noqa: BLE001
                    continue
        return out

    # MATCHED sample sizes. Distinct-R-group counts scale with how many
    # molecules you decompose, so comparing 1,200 unfiltered against 192
    # filtered would show a difference that is pure sample size.
    k = min(args.decompose, len(inside), len(mols))
    print(f"   (comparing {k:,} molecules from each pool -- matched)\n")
    for label, pool in (("all ZINC", mols), ("envelope-filtered", inside)):
        take = pool[:k]
        if not take:
            continue
        seen: Counter = Counter()
        for m in take:
            seen.update(hashes_of(m))
        if not seen:
            print(f"   {label}: nothing decomposed")
            continue
        distinct = set(seen)
        shared = distinct & set(lib)
        # what share of the FLAVOUR library's occurrence mass does ZINC reach?
        mass = sum(lib[h] for h in shared) / lib_total
        # and what share of ZINC's own R-group occurrences are already known?
        zinc_known = sum(seen[h] for h in shared) / sum(seen.values())
        print(f"   {label:20s} n={len(take):,}  distinct R-groups {len(distinct):,}")
        print(f"     {'shared with flavour library':32s} {len(shared):,} "
              f"({100*len(shared)/len(distinct):.1f}% of ZINC's)")
        print(f"     {'flavour-library mass reached':32s} {100*mass:.1f}%")
        print(f"     {'ZINC occurrences already known':32s} {100*zinc_known:.1f}%")
    print("\n   'mass reached' is the number that matters: the flavour library is")
    print("   heavily skewed, so distinct-count overlap understates how much of the")
    print("   actual retrieval target space ZINC pretraining would touch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
