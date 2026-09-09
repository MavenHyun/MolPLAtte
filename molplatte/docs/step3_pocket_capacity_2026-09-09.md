# STEP 3 — why pocket conditioning is inert

Runs of 2026-09-09, seed 911012, corpus `tastepocket_corpus` (243 records),
scored against the 91,935-row union library (effective size 946). Folds are
connected components of the (ligand, receptor) graph, so a held-out receptor is
unseen. Fold 1 holds out TRPV1+TRPA1 entirely and is excluded from aggregates,
as in [step2_pocket_results_2026-09-05.md](step2_pocket_results_2026-09-05.md).

## Headline

**The pocket representation is informative. The task cannot use it.**

The 2026-09-05 null had two candidate explanations needing opposite fixes: too
few receptors, or a representation that destroys the signal. Both are now ruled
out, and the actual cause is a **granularity mismatch** — the pocket resolves
*chemotype*, R-group retrieval demands *exact fragment identity*.

| mechanism | trainable params | effect |
|---|---:|---:|
| non-parametric kNN over pocket similarity | 0 | +0.016 H@1 vs prior (1.8 SE) |
| fixed PCA-32 basis + adapter | 1,056 | +0.002 H@1 vs free (0.13 SE) |
| free reduction 1280→128→32 (Sept) | 168,096 | +0.0008 H@1 vs pocket-off |

Capacity spans zero to 168,096 parameters and every arm is null. **That axis is
exhausted.**

---

## 1. The representation carries real signal

A no-training probe over the 1,255 pocket sites: for each held-out pocket, take
the nearest other pocket by ESM-2 cosine and compare its ligand to the true one
by Morgan/Tanimoto. Splits are receptor-disjoint.

First pass gave 0.5301 vs 0.1273 — inflated, because 1,255 sites share only 184
ligands and the neighbour was *the same compound in a homolog* 32.5% of the
time. Excluding every training pocket bound to the held-out ligand:

| condition | nearest pocket | random (prior) | delta |
|---|---:|---:|---:|
| different receptor, **unseen ligand** | 0.3505 | 0.1228 | +0.228 (22.9 SE) |
| + **different family** | 0.1858 | 0.0953 | +0.091 (13.1 SE) |

So the pooled ESM-2 vector predicts what chemistry a pocket binds, across
receptors it has never seen and even across family boundaries. PCA-32 preserves
this (+0.2145, cross-family +0.0968) while retaining 97% of variance, so the
reduction is not where anything is lost.

## 2. But it is worth nothing on the actual task

Rank library R-groups for a held-out joint using only the pocket: a
similarity-weighted vote over the k nearest training pockets, blended with the
frequency prior.

A hard kNN scores *below* the prior at H@10, but that is the ranking rule, not
the pocket: it front-loads ~30 candidates and pushes everything else down, so it
loses by construction whenever the target is not among them. Blending fixes it
and gives H@1 0.1090, MRR +0.0505 over the prior.

**That number is selection bias.** K, temperature and blend weight were chosen on
the same folds being scored — 60 configurations against 670 targets. With nested
selection, each outer fold choosing its configuration from the other folds only:

| | pocket kNN | prior | delta |
|---|---:|---:|---:|
| H@1 | 0.0597 | 0.0433 | +0.0164 (1.8 SE) |
| H@10 | 0.3030 | 0.3194 | −0.0164 (1.3 SE) |
| MRR | 0.1342 | 0.1201 | +0.0141 (1.9 SE) |

Nothing clears 2 SE. Two of four folds picked configurations no better than the
prior on their own held-out data; fold 4's inner CV chose blend weight 0.0, i.e.
"ignore the pocket".

**This is the key measurement.** It uses the pocket metric structure directly,
non-parametrically, with nothing to overfit — an upper bound on what any method
ranking by pocket similarity can achieve. Tanimoto 0.35 means "roughly the right
kind of molecule"; the R-group vocabulary is keyed on exact WL hashes. Coarse
chemotype does not resolve to exact fragments.

## 3. The capacity hypothesis, tested and refuted

The free reduction is 168,096 parameters fitted from 243 records — 692 per
record. The hypothesis was that a reduction that underdetermined cannot preserve
metric structure and instead fits receptor identity, which is useless on a
held-out receptor. `PocketConditioning(basis_path=...)` replaces it with a frozen
per-fold PCA projection plus a 32→32 adapter: **1,056 parameters, 161× fewer**.

| fold | N | free H@1 | PCA H@1 | delta |
|---|---:|---:|---:|---:|
| 0 | 178 | 0.1180 | 0.1124 | −0.0056 |
| 1 * | 197 | 0.0558 | 0.0406 | −0.0152 |
| 2 | 141 | 0.1631 | 0.1915 | +0.0284 |
| 3 | 147 | 0.1197 | 0.1224 | +0.0027 |
| 4 | 138 | 0.2536 | 0.2391 | −0.0145 |

Unseen-receptor aggregate: H@1 **0.1603 → 0.1622 (+0.0020, 0.13 SE)**, MRR
0.2445 → 0.2416. Per-fold deltas swing −0.015 to +0.028 with mixed signs, which
is what noise looks like at ~150 targets per fold.

Constraining capacity 161-fold changes nothing. Combined with §2, the pocket
contributes nothing at 0, 1,056, or 168,096 parameters.

### The basis is fitted per fold, without the fold

PCA directions describe the variance of the data they see, so one basis over all
pockets would encode the held-out receptors and the fold would stop measuring
generalisation. `build_pocket_basis.py` writes `fold{k}.npz` from that fold's
training records only, and `full.npz` for the final model alone. Orthonormality
is asserted, not assumed (max |CC^T − I| = 2.0e-15).

## 4. What DOES move Step 3: the warm start

The first PCA sweep was launched with `STEP1=contrain_coconut-flavordb-full`
while the September baseline used the script default
`s1-pretrain-full-s911012`. Both expand to the same filename, so nothing
complained; the load lines differ (141/141 with 0 unexpected against 141/157
with 16, the extra being an assembly head). Rerun with the warm start matched:

| arm | H@1 | MRR |
|---|---:|---:|
| PCA basis, **ZINC→flavour** warm start | 0.0897 | 0.1746 |
| PCA basis, **flavour-only** warm start | 0.1622 | 0.2416 |

**+0.0726 H@1, 4.84 SE — 36× the basis effect**, from a choice that has nothing
to do with pockets. It matches the same day's finding that ZINC→flavour is the
worse encoder ([loss_reweighting_and_freezing](loss_reweighting_and_freezing_2026-09-09.md)),
and it is the actionable lever for Step 3.

The confounded run is kept as `*-pca-zincwarm*`: it is a valid warm-start
measurement, just not a basis measurement.

## 5. What this licenses

Supported:
- Pocket conditioning as formulated — pooled ESM-2, condvec concatenation,
  R-group hash retrieval — contributes nothing, and the cause is granularity,
  not data volume and not reduction capacity.
- The ESM-2 pocket space genuinely encodes ligand chemotype across unseen
  receptors.

NOT supported:
- That pockets are useless for lead optimisation. Chemotype-level signal is real
  and might serve a coarser objective (filtering candidates, ranking whole
  molecules) even though it cannot pick exact fragments.
- Any claim about 3D geometry, which remains untested.

## 6. Next

**The EGNN question is now genuinely open rather than deprioritised.** The
09-05 reasoning — geometry cannot be the bottleneck while the sequence signal is
injected and does nothing — no longer holds, because we now know the sequence
signal is chemotype-coarse. Whether geometry resolves finer contacts is exactly
the untested question.

Test it cheaply first: `pocket_probe2.py` scores any pocket representation in
about a minute with no training. An EGNN embedding should clear the sequence
baseline (+0.228 unseen-ligand, +0.091 cross-family) before any training run is
committed. If it does not, it will not help retrieval either.

The second option is to change the target rather than the encoder: a coarser
retrieval unit (chemotype cluster instead of exact WL hash) matches the
granularity the pocket actually resolves.

## Reproducing

```bash
python3 molplatte_preprocess/scripts/build_pocket_basis.py     # per-fold bases
BASIS_DIR=/home/mogan/preprocessed/molplatte/pocket_basis \
  STEP1=$CK/s1-pretrain-full-s911012_best.pt \
  bash molplatte/src/scripts/run_step2_pocket_cv.sh
```

wandb: project `molplatte`, group `step2-pocket-cv`.
