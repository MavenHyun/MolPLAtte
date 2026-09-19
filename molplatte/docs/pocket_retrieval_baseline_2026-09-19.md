# Against a frequency baseline, the pocket wins only in the top-1..5 regime

The granularity probe reported nearest-pocket R-group retrieval at **15x
enrichment**, measured against a RANDOM-POCKET baseline. That was the wrong
competitor. The tastepocket targets sit in the HEAD of the frequency
distribution -- 0.286% of library entries carrying **48.6%** of the prior mass,
typical target seen ~1,200 times in 393k records -- so a constant "predict the
commonest fragments" rule is already strong, and beating random does not show
the pocket is worth consulting.

Re-run against that competitor, at matched budget k:

| k | FREQ | RANDPKT | NNPKT | NN−RAND | σ | **NN−FREQ** | σ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0436 | 0.0048 | **0.1174** | +0.1125 | 11.0 | **+0.0737** | 6.3 |
| 5 | 0.1455 | 0.0223 | 0.1756 | +0.1532 | 12.7 | +0.0301 | 2.2 |
| 10 | 0.2561 | 0.0504 | 0.2396 | +0.1891 | 13.9 | -0.0165 | -1.3 |
| 25 | 0.2842 | 0.1339 | 0.3143 | +0.1804 | 13.3 | +0.0301 | 2.5 |
| 50 | 0.2949 | 0.2231 | 0.3492 | +0.1261 | 10.2 | +0.0543 | 4.5 |
| 100 | **0.5703** | 0.3152 | 0.3744 | +0.0592 | 6.2 | **-0.1959** | -10.8 |

(hit@k; recall@k tracks it and ties at k=100: 0.2799 vs 0.2806.)

`RANDPKT` is the frequency-matched control -- union of RANDOM pockets' R-groups
grown to k by the same rule as `NNPKT`, so it shares the generative process and
therefore the frequency profile, differing only in that its pockets are not
chosen by similarity.

## Two separate conclusions

**Pocket similarity carries real signal.** `NNPKT - RANDPKT` is +0.11 to +0.19
hit@k at 6-14σ across every budget. This is not an artifact of common fragments.

**As a predictor it beats frequency only at small k.** At k=1 nearest-pocket is
2.7x better (0.1174 vs 0.0436). By k=10 the two cross, and at k=100 frequency
wins by -0.1959 (-10.8σ). The cause is structural: ligands carry ~1.23 R-groups
each, so growing a nearest-pocket prediction to k=100 drags in ever more distant
pockets, while frequency at k=100 simply covers the head holding 48.6% of the
mass.

## Why this explains the training null better than anything prior

The model already scores with `sim/τ + coef·log p(k)` -- **the frequency prior
is in the objective**. The pocket's marginal value *over that prior* is roughly
+0.03 hit@5. That is the quantity 269 records would have to teach a contrastive
objective to extract. It is small enough to account for every null measured --
eight rung-vs-twin comparisons, two learning rates, two initialisation scales --
without needing catastrophic forgetting, target granularity, or any defect in
the pocket preprocessing (which the ProLIF run independently validated: 1,247 of
1,254 pockets make chemically coherent contacts with their own crystal ligand).

## What to build, and what not to

Wire the nearest-pocket set in as a **top-1..5 re-ranking prior** over the
existing logQ-corrected score. Do not use it as a retrieval replacement: past
k~10 it is worse than a constant frequency guess, so a pocket-driven candidate
list would be actively harmful at the list sizes the notebook renders.

## Still open

- Single seed (911012) throughout the pocket work.
- `val` and `test` are the same held-out fold in the CV.
- 17 families / 96 receptors; TRPV1 alone is 522 of 1,255 pockets.

## Reproduce

```bash
python molplatte/src/scripts/pocket_retrieval_baseline.py
```
