# Pockets DO predict R-groups. The target-granularity hypothesis is refuted.

**This overturns the explanation carried through the EGNN, ProLIF and
forgetting writeups.** Those documents each closed by suggesting the remaining
problem was the retrieval target -- that an R-group keyed by a WL subgraph hash
cannot resolve individual contacts, so a pocket has nothing to condition on.
Measured directly, that is false. Pocket similarity predicts R-groups **as
strongly as it predicts whole ligands**, at the exact granularity the model
retrieves at, and the signal survives a receptor-family boundary.

## Result

Same protocol as every other pocket probe (nearest OTHER pocket by cosine,
different receptor AND different ligand, delta against a random pocket from the
same pool). Only the scoring TARGET changes. 1,031 pockets joined to a
decomposition.

| target | nearest | random | delta | SE | cross-family | SE |
|---|---:|---:|---:|---:|---:|---:|
| ligand *(published control)* | 0.2910 | 0.1128 | +0.1782 | 18.3 | +0.0635 | 8.8 |
| core / scaffold | 0.2610 | 0.1106 | +0.1505 | 17.4 | +0.0567 | 8.8 |
| **R-group (best-match Tanimoto)** | 0.3344 | 0.1558 | **+0.1786** | **20.8** | **+0.0853** | **10.9** |
| **R-group (exact WL hash, Jaccard)** | 0.0632 | 0.0041 | **+0.0591** | 10.3 | +0.0277 | 8.7 |

- R-groups reach **100.2%** of the ligand-level delta, at *higher* significance
  (20.8σ vs 18.3σ).
- R-groups have the **strongest cross-family transfer of any target** (+0.0853,
  10.9σ) -- stronger than the whole ligand.
- At exact WL-hash granularity the nearest pocket's R-group set overlaps the
  true one at 0.0632 against a random baseline of 0.0041: a **15x enrichment**
  at 10.3σ.

## Two ways this nearly went wrong

**The `core` row first returned n=0.** `core_smiles` is empty for all 734
decompositions in this corpus; only `core_atoms` is populated. Reported without
checking, a storage gap would have become the finding "pockets say nothing
about the scaffold".

**R-groups are not whole molecules.** If the decomposition had left the ligand
intact, `rgroup-tan ≈ ligand` would be a tautology. It does not: R-groups cover
**37% of the parent on average** (p10 0.17, median 0.38, p90 0.58), and 0 of 734
decompositions cover the whole molecule.

## What this changes

1. **Retract the target-granularity explanation** from
   `egnn_pocket_probe_2026-09-09.md`, `ifp_pocket_probe_2026-09-17.md` and
   `pocket_forgetting_2026-09-17.md`. The information the model needs is
   present in the pocket representation at the granularity it retrieves at.
2. **The gap is extraction, not availability.** The finetuning null stands as
   measured -- 8 rung-vs-twin comparisons, all null -- but its cause is now
   narrowed: a contrastive objective on 269 records does not recover signal that
   a plain nearest-neighbour lookup finds at 10σ.
3. **A non-parametric pocket retrieval is now the obvious move, and it needs no
   training.** Take the query receptor's pocket embedding, find the nearest
   pocket in the corpus, propose its R-groups as priors. That already delivers
   15x enrichment on exact hash overlap and +0.0853 cross-family on fragment
   similarity. It is also the baseline any trained pocket model must beat, and
   no trained model here has.

## Still open

- Single seed (911012) throughout the pocket work; no seed replication.
- `val` and `test` are the same held-out fold, so early stopping selected on
  scored data (shared by all arms; absolute numbers optimistic).
- 17 families / 96 receptors, one family (TRPV1) is 522 of 1,255 pockets --
  cross-family numbers rest on few families.

## Reproduce

```bash
python molplatte/src/scripts/rgroup_granularity_probe.py
```
