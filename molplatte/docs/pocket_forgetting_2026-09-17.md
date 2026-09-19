# The step-3 pocket result was catastrophic forgetting. Correcting it turns "harmful" into "inert".

> **PARTLY RETRACTED (2026-09-19).** The closing suggestion that the
> RETRIEVAL TARGET is the remaining problem is refuted: pockets predict
> R-groups as well as whole ligands (+0.1786 vs +0.1782), with 15x
> enrichment on exact WL-hash overlap. See
> [rgroup_granularity_probe_2026-09-19.md](rgroup_granularity_probe_2026-09-19.md).

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

## Follow-up: the full capacity ladder (added same day)

The first pass tested only the extremes -- 1,056 params (inert) and 2.1M
(destructive) -- and reported a null. That was under-tested: the configuration
most likely to show a positive, a trainable `query_projector` (the module that
actually consumes the pocket vector), had not been run. Every rung below now
has its own permuted-pocket twin, so information and capacity are separated at
each capacity level rather than only at the bottom.

| rung | trainable | H@1 | vs base | H@10 | **vs its TWIN** |
|---|---:|---:|---:|---:|---:|
| baseline | — | 0.1715 | — | 0.3930 | — |
| pocket only | 1,056 | 0.1715 | +0.0000 | 0.3941 | +0.0011 (+1.0σ) |
| query_projector + pocket | 556,064 | 0.1510 | -0.0205 | 0.4058 | **+0.0000 (=)** |
| all projectors + pocket | 2,135,072 | 0.1350 | -0.0365 | 0.4285 | **+0.0000 (=)** |
| everything | 19.7M | 0.1448 | -0.0267 | 0.4299 | -0.0011 (-1.0σ) |

**Every rung ties its own twin.** H@10 climbs steadily with capacity
(0.3930 -> 0.4299) while H@1 falls (0.1715 -> 0.1350) -- but the shuffled twins
climb and fall by exactly the same amounts. The whole capacity effect is
capacity; none of it is pocket information. At the two middle rungs the real and
permuted arms produce *different weights* (checkpoint hashes differ) yet
*identical rankings* -- MRR matches to four decimals across all five folds.

### Mechanism: the pocket columns are 40x weaker than flavour and never grow

`query_projector` takes `[hidden 512 | flavour 24 | pocket 32]`. Mean |w| by
block:

| checkpoint | hidden | flavour | **pocket** | pocket/flavour |
|---|---:|---:|---:|---:|
| base (no finetune) | 0.1905 | 0.8281 | 0.0209 | 0.025 |
| pocket only | 0.1905 | 0.8281 | 0.0209 | 0.025 |
| query_projector + pocket | 0.1905 | 0.8281 | 0.0210 | 0.025 |
| everything (19.7M trainable) | 0.1905 | 0.8281 | 0.0211 | 0.026 |

The pocket block starts ~40x weaker than the flavour block and moves **<1% even
when the entire 19.7M-parameter network is unfrozen**. The pocket cannot
meaningfully reach the query embedding, which is why permuting it changes
nothing.

**This qualifies the null rather than strengthening it.** "Inert" is solid as an
empirical result -- it holds across four capacity levels, each against its own
control. But the mechanism is an initialisation-scale problem as much as an
information problem: a pocket block this weak at init cannot discover signal
even if signal existed, so the experiment is under-powered to prove *absence* of
pocket information. The distinguishing experiment is to rescale the pocket block
(or give it a separate, larger learning rate) and re-run the ladder; if it still
ties its twins with the blocks at comparable magnitude, the null is mechanistic
rather than an artifact.

## Rescaling the pocket block: the caveat resolves, and my mechanism was wrong

The section above closed by flagging that the pocket block sat 40x weaker than
the flavour block, so the null might be an initialisation artifact rather than
an absence of signal, and proposed rescaling as the distinguishing test. The
rescale was run (`scripts/rescale_pocket_block.py`, x39.62 to flavour parity)
and the whole ladder repeated. **The proposed mechanism was wrong, and the
caveat dissolves for a different reason than expected.**

The stated reasoning was: the gradient reaching the zero-init adapter is
proportional to the query projector's pocket columns, so weak columns starve the
adapter. That is true of SGD. **This model trains with Adam**
(`lightning_modules/base.py:103`), which divides the update by the gradient's
own RMS and is therefore *invariant to gradient rescaling*. Multiplying the
columns by 39.62 multiplies the adapter's gradient by 39.62 and its update by 1.

Measured, and unambiguous:

| rung | adapter max\|w\|, scale 1 | adapter max\|w\|, scale 39.6 | ratio |
|---|---:|---:|---:|
| pocket only | 0.000685 | 0.000684 | **1.0x** |
| query_projector + pocket | 0.000690 | 0.000688 | **1.0x** |
| all projectors + pocket | 0.001044 | 0.001040 | **1.0x** |
| everything | 0.000912 | 0.000920 | **1.0x** |

The Adam signature is visible directly in the scale-1 ladder: `max|w|/lr` is
6.8 at lr 1e-4 and 6.9 at lr 1e-3 -- the adapter moves a fixed number of
*learning-rate-sized* steps regardless of how large the gradient is.

### The intervention that does work was already run

Adam is invariant to gradient scale but **not** to learning rate. The lever I
should have reached for was lr, and `adapter-lr1e-3` had already pulled it: a
10x larger adapter excursion (max\|w\| 0.000685 -> 0.006851). It ties its twin
and the baseline exactly (+0.0000 H@1). So the "the adapter was too small to
find out" objection was already answered before the rescale was run.

### The rescaled ladder as independent replication

| rung | params | scale 1: ΔH@1 vs twin | scale 39.6: ΔH@1 vs twin | scale 39.6: ΔH@10 vs twin |
|---|---:|---:|---:|---:|
| pocket only | 1,056 | +0.0000 (=) | -0.0014 (-1.0σ) | +0.0039 (+1.4σ) |
| query_projector + pocket | 556,064 | +0.0000 (=) | +0.0014 (+1.0σ) | +0.0000 (=) |
| all projectors + pocket | 2,135,072 | +0.0000 (=) | -0.0032 (-1.5σ) | -0.0011 (-1.0σ) |
| everything | 19.7M | +0.0025 (+1.6σ) | +0.0014 (+1.0σ) | +0.0025 (+1.6σ) |

Eight rung-versus-twin comparisons across the two ladders. Every one is within
+-0.0032 H@1, none exceeds 1.6σ, and the signs are mixed -- the distribution of
a quantity whose true value is zero.

The rescale was also verified prediction-neutral before use: because
PocketConditioning's output is zero-init, the pocket slice is exactly zero at
initialisation and these columns multiply zero. The rescaled baseline reproduces
the original on all five folds (0.1803 / 0.1026 / 0.1773 / 0.1631 / 0.2340).

### Where this leaves it

The initialisation-scale caveat is retired: under Adam the block magnitude
cannot be the limiter, and the lever that *can* change adapter movement (lr) was
tested at 10x with a null. What actually bounds the adapter is that the pocket
never improves validation loss, so the best checkpoint is selected within the
first few epochs -- and that is the finding, not an artifact of it.

The honest residual is narrow: no run trained past early stopping with a
pocket-only warm-up. On 269 records that is far more likely to overfit than to
reveal signal, and it would have to overturn a consistent result from four
independent directions -- sequence embeddings, 3D geometry, typed interactions,
and supervised finetuning at four capacity levels against matched controls.

## Reproduce

```bash
python molplatte/src/scripts/run_pocket_forgetting_cv.py --gpu 1 --folds 0,1,2,3,4

# rescaled ladder
python molplatte/src/scripts/rescale_pocket_block.py \
  ~/checkpoints/molplatte/exp-wide512-pocket.pt \
  --out ~/checkpoints/molplatte/exp-wide512-pocket-rescaled.pt
python molplatte/src/scripts/run_pocket_forgetting_cv.py --gpu 1 --folds 0,1,2,3,4 \
  --base ~/checkpoints/molplatte/exp-wide512-pocket-rescaled.pt --prefix pfr- \
  --results ~/checkpoints/molplatte/pocket_rescaled_results.csv
# -> ~/checkpoints/molplatte/pocket_forgetting_results.csv
```
