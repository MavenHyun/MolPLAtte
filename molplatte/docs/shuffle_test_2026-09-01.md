# Condvec shuffle test — 2026-09-01

Rerun after the condvec fix (`24b75d1`). The previous run (2026-08-31) is void:
the condition vector then collapsed to a single `MW > 350` bit, so its "real
condition" arm carried no flavour at all.

**Setup.** `coconut-flavordb-filtered` (build tag `r333-m2-h4-flavor24v2`),
25 epochs, logQ on, 3 arms x 3 seeds (911012 / 20260901 / 424242) on GPU1.
Filtered was chosen over `-full` because its label density is 23.3% against
2.6%, i.e. ~9x the signal per unit of compute.

Three arms isolate two different things:

| arm | condvec | what it measures |
|---|---|---|
| `nocond` | absent | baseline |
| `shuf` | permuted across molecules | capacity of a 24-wide input, no information |
| `cond` | real | capacity + information |

`cond - shuf` is the information; `shuf - nocond` is the capacity.

## Corpus-wide (val H@1, N=7,226)

| arm | mean | sd |
|---|---:|---:|
| nocond | 0.3169 | 0.0040 |
| shuf | 0.3220 | 0.0026 |
| cond | 0.3255 | 0.0053 |

Paired by seed:

```
cond - nocond  (total)        +0.0086   t(2)=+7.45   p≈0.018
shuf - nocond  (capacity)     +0.0051   t(2)=+2.32   p≈0.146
cond - shuf    (INFORMATION)  +0.0035   t(2)=+1.04   p≈0.406
```

The total effect is real but **60% of it is capacity, not flavour**. The
information component is not separable from noise at n=3 — the minimum
detectable effect here is 0.0143, and resolving an effect of 0.0035 would take
roughly 52 seeds.

## Split by whether the query actually has a flavour label

Only ~25% of queries carry a real sensory class; the rest are `odorless` or
`unknown`. Averaging over all of them dilutes any effect ~4x. Splitting the
prediction tables (last epoch, per seed, paired):

| subset | nocond | shuf | cond | cond − shuf | p |
|---|---:|---:|---:|---:|---:|
| **has a real flavour label** | 0.3774 | 0.3531 | **0.4156** | **+0.0625** | ≈0.061 |
| no label (`odorless`/`unknown`) | 0.3019 | 0.3067 | 0.3032 | −0.0036 | ≈0.726 |

Per-seed `cond − shuf` on the labelled subset: `+0.031, +0.074, +0.083` —
same sign every seed, and `cond` has the tightest spread of any arm (sd 0.0072).
On the unlabelled subset the sign flips across seeds and the mean is ~0.

## Reading

1. **The condvec is not leaking.** A vector that encoded the target would lift
   retrieval everywhere. This one does nothing (−0.004) where it carries no
   information, which is the specificity a leak cannot fake.
2. **It does appear to help where it has something to say** — +6.2 points of
   H@1 on labelled queries, consistent across seeds.
3. **The corpus-average metric hides this**, because three quarters of the
   corpus has no label to condition on. Corpus-wide H@1 is the wrong primary
   metric for conditioning work.

## Caveats

`p≈0.061` at n=3 is not significance, and the label split is a secondary,
post-hoc analysis computed from a 500-query-per-epoch sample rather than the
full 7,226-query validation set — so the two tables are not directly
comparable in magnitude. Confirming this needs more seeds and a retrieval
metric computed on the labelled subset directly, not sampled tables.

Reproduce: `molplatte/src/scripts/run_shuffle_test.sh`
(`CORPUS=… EPOCHS=… SEEDS=… GPU=… CONC=…`).
