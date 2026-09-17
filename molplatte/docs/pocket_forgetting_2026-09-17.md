# The step-3 pocket result was catastrophic forgetting. Correcting it turns "harmful" into "inert".

**Both halves of this matter.** The September finding -- pocket finetuning costs
-0.0763 H@1 on unseen receptors -- was **confounded**: that run left 2.1M
parameters adapting to 269 records at the sweep's *worst* learning rate. When
the flavour pathway is frozen so forgetting is impossible, the loss disappears
entirely (exactly 0.0000 on all five folds).

**But the corrected experiment does not rescue pocket conditioning.** It shows
the pocket is *inert*, not useful: a model trained on **real** pocket embeddings
and one trained on **randomly permuted** pocket embeddings reach identical
validation loss to four decimal places on every fold.

## Why the original run could not answer the question

`w512-pocket-fold0..4` used `freeze=[encoder]` at `lr=1e-3`:

- the graph encoder (16.9M params) was frozen, but the **projectors -- 2.1M
  trainable params, including the `query_projector` that actually consumes the
  pocket vector -- were not**;
- `lr=1e-3` is the worst setting in the width-512 sweep (0.3411 vs 0.3700 at
  1e-4), and the freeze x lr interaction *reverses* across that range.

Catastrophic forgetting and "pockets do not help" predict the same number, so
the -0.0763 could not distinguish them.

## Design: make forgetting impossible rather than measure it

`freeze=[encoder, projectors]` leaves **only `pocket_conditioning` trainable --
1,056 parameters** behind a frozen PCA basis with a zero-initialised output. The
model therefore starts numerically identical to the no-finetune baseline, and
the flavour pathway *cannot* move. Verified at the weight level, not assumed:

| arm | trainable params | projector drift vs base |
|---|---:|---:|
| `enc-*` | 2,135,072 | ~0.5 |
| `adapter-*` | 1,056 | **0.000e+00** |

Baseline is re-derived through the same eval path, never inherited from the
0.1882 in the notes. `val` and `test` are the same held-out fold
(`data_modules/base.py:374`), so `run_mode=test` scores exactly the held-out
receptors.

## Results: 5 folds, 801 held-out queries, paired per-fold deltas

| arm | freeze | lr | H@1 | ΔH@1 | H@10 | ΔH@10 | MRR | ΔMRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (no finetune) | — | — | 0.1715 | — | 0.3930 | — | 0.2489 | — |
| **`enc-lr1e-3`** *(the original)* | encoder | 1e-3 | 0.1171 | **-0.0543 (-7.3σ)** | 0.3470 | -0.0460 (-1.4σ) | 0.1973 | **-0.0516 (-3.3σ)** |
| `enc-lr1e-4` | encoder | 1e-4 | 0.1350 | -0.0365 (-1.3σ) | 0.4285 | +0.0355 (+1.2σ) | 0.2330 | -0.0159 (-0.6σ) |
| **`adapter-lr1e-4`** | enc+proj | 1e-4 | 0.1715 | **+0.0000** | 0.3941 | +0.0011 (+1.0σ) | 0.2489 | -0.0001 |
| `adapter-lr1e-3` | enc+proj | 1e-3 | 0.1715 | +0.0000 | 0.3952 | +0.0022 (+1.0σ) | 0.2489 | -0.0000 |
| `adapter-shuf` *(pocket permuted)* | enc+proj | 1e-4 | 0.1715 | +0.0000 | 0.3930 | +0.0000 | 0.2488 | -0.0001 |

Reading down the ΔH@1 column: **-0.0543 -> -0.0365 -> 0.0000** as the flavour
pathway is progressively protected. The harm tracks how much of the network was
allowed to move, not whether pockets were used.

## The decisive control: real pockets train identically to shuffled ones

`shuffle_pocket_only` permutes the pocket half across molecules and leaves the
flavour bits untouched (the pre-existing `shuffle_condvec` permutes all 1,304
columns and destroys flavour too, so it could not isolate this). Verified before
use: flavour identical in 80/80 samples, pocket changed in 80/80.

Best `val/loss`, real pocket vs permuted pocket:

| fold | real | permuted | Δ |
|---|---:|---:|---:|
| 0 | 8.4352 | 8.4356 | 0.0004 |
| 1 | 7.4285 | 7.4284 | 0.0001 |
| 2 | 8.1587 | 8.1583 | 0.0004 |
| 3 | 7.3938 | 7.3940 | 0.0002 |
| 4 | 7.6398 | 7.6400 | 0.0002 |

Identical checkpoint-save counts per fold (4/3/3/1/4 in both arms) and adapter
weights matching to three significant figures. **The optimiser cannot tell a
real pocket from a shuffled one.**

Two corroborating details:

- Adapter `max|w|` scales *exactly* with the learning rate -- 0.00101 at 1e-4,
  0.010101 at 1e-3, a clean 10x. The weights are integrating a near-constant
  tiny gradient rather than converging to anything, which is what a flat loss
  surface in the pocket direction looks like.
- On fold 3 the adapter never improved on its epoch-0 score at all, so
  `max|w|` stayed at exactly 0.0.

## What this changes

1. **Retract the attribution, not the number.** "Pocket conditioning is harmful
   at width 512" was wrong. The correct claim is: *the step-3 finetuning recipe
   was harmful; pocket conditioning itself is inert.* The shipped
   `molplatte-final-v2-w512.pt` is unaffected -- it never included step 3.
2. **A step-3 checkpoint is now defensible, and still pointless.** The adapter
   recipe costs nothing (0.0000 on every fold), so shipping one would no longer
   ship a regression. It would also add nothing.
3. **Converges with the two probes.** EGNN (2026-09-09) and ProLIF
   (2026-09-17) both found no pocket representation beating ESM-2, and ProLIF
   found no cross-family signal even from an oracle. This experiment shows the
   same thing from the training side: the gradient cannot use the pocket.
   Sequence, 3D geometry, typed interactions, and now supervised finetuning all
   land in the same place.

The remaining hypothesis worth testing is not a better pocket encoder but the
**retrieval target**: an R-group keyed by WL hash cannot resolve individual
contacts, so there may be nothing for a pocket to condition.

## Reproduce

```bash
python molplatte/src/scripts/run_pocket_forgetting_cv.py --gpu 1 --folds 0,1,2,3,4
# -> ~/checkpoints/molplatte/pocket_forgetting_results.csv
```
