# Hyperparameter sweep: width and learning rate were never examined

Staged sweep across s1 → s2, 2026-09-12/15, seed 911012 unless stated.
All arms scored through ONE path — `run_mode=test` on `coconut-flavordb-full`
against the 668,913-row `zincbase__flavor__crossdocked__tastepocket` library,
prior 0.0233 on every arm, which is how you can tell the query sets match.

## Headline

**H@1 0.3098 → 0.3701 (+19%). novel@10 0.3401 → 0.4236 (+25%).**

Both gains come from two parameters that had never been swept in this project:
`hidden_dim` and `learning_rate`, inherited unexamined from MolPLA's release.
Every earlier conclusion here was drawn on a model that was capacity-starved
and trained at roughly 10x the right step size.

| | H@1 | H@10 | MRR | novel@10 |
|---|---:|---:|---:|---:|
| MolPLA defaults (300, lr 1e-3) | 0.3098 | 0.6109 | 0.4117 | 0.3401 |
| **tuned (512, lr 1e-4, no freeze)** | **0.3701** | **0.6711** | **0.4725** | **0.4236** |

---

## 1. Width: a clean scaling curve, saturating

| hidden | params | H@1 | H@10 | MRR | novel@10 | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 6.6 M | 0.3098 | 0.6109 | 0.4117 | 0.3401 | — |
| 384 | 10.7 M | 0.3229 | 0.6293 | 0.4258 | 0.3497 | +0.0131 |
| **512** | 19.0 M | 0.3465 | 0.6482 | 0.4490 | 0.3758 | +0.0367 |
| 768 | 42.7 M | 0.3604 | 0.6568 | 0.4610 | 0.3918 | +0.0506 |
| 1024 | 75.8 M | 0.3685 | 0.6671 | 0.4703 | 0.4143 | +0.0587 |

`H@1 ≈ 0.0170·log2(params) − 0.0712`, R² 0.963, residuals negative at the top —
saturating. Marginal return per doubling: +0.0185, **+0.0286**, +0.0119, +0.0098.

**512 is the knee**: 63% of the total gain for 2.9x the parameters, against
1024's 100% for 11.5x. Everything downstream uses it.

## 2. Depth: monotonic, and it refutes the oversmoothing lead

| conv layers | H@1 | Δ | oversmoothing |
|---:|---:|---:|---:|
| 3 | 0.2928 | −0.0170 (−5.2 SE) | 0.314 |
| 5 (default) | 0.3098 | — | 0.331 |
| 7 | 0.3180 | +0.0082 (+2.5 SE) | 0.249 |

[representation_health](representation_health_2026-09-10.md) recorded MAD
declining 33–41% across layers and offered shallower-may-help as its one
architectural lead. **Wrong**: shallower is worse by 5 SE. And oversmoothing
does not predict retrieval at all — `conv7` has the LOWEST (0.249) and
`wide512` the HIGHEST (0.441), and `wide512` wins by a wide margin.

## 3. Learning rate: the dominant axis, and it hid a confound

τ ∈ {0.01, 0.05} × lr ∈ {1e-3, 3e-4} × freeze ∈ {none, encoder} on `wide512`,
then lr extended once the grid turned out to have found an edge rather than an
optimum.

| lr | freeze=none | freeze=encoder |
|---|---:|---:|
| 1.0e-3 | 0.3411 | 0.3532 |
| 3.0e-4 | 0.3647 | 0.3611 |
| **1.0e-4** | **0.3701** | 0.3598 |
| 3.0e-5 | 0.3670 | — |

Marginal means: **lr** 3e-4 0.3565 vs 1e-3 0.3456 (+0.0109, dominant) ·
**τ** 0.01 0.3550 vs 0.05 0.3471 (the default was already right) ·
**freeze** 0.3513 vs 0.3508 (+0.0005, gone).

### The freeze effect is an lr artifact

At 1e-3 freezing HELPS (+0.0121). At 1e-4 it HURTS (−0.0103). Freezing was
compensating for too large a step size, and
[loss_reweighting_and_freezing](loss_reweighting_and_freezing_2026-09-09.md)'s
+0.0098 is that compensation measured in isolation and read as a property of
freezing. That document is amended.

## 4. Seeds: the variance is smaller than assumed

Three seeds per config (911012, 20260914, 4242):

| config | mean | sd |
|---|---:|---:|
| none / 1e-4 | **0.3701** | 0.0009 |
| none / 3e-4 | 0.3672 | 0.0021 |
| none / 3e-5 | 0.3662 | 0.0013 |
| encoder / 1e-4 | 0.3598 | 0.0005 |

| comparison | Δ | seed SE | |
|---|---:|---:|---|
| 1e-4 vs 3e-4 | +0.0029 | 0.0013 | +2.2σ |
| 1e-4 vs 3e-5 | +0.0039 | 0.0009 | +4.3σ |
| freeze none vs encoder | +0.0103 | 0.0006 | **+17.8σ** |

**Seed sd (0.0005–0.0021) is SMALLER than the query-sampling SE (0.0033).** I
had assumed the opposite and had been quoting the query SE as the conservative
bound; for these comparisons it is the loose one.

## 5. ZINC is unnecessary

`sw-s1-flavour-assembly` — flavour-only, assembly head enabled — gives
**identical retrieval** (H@1 0.3098) and an assembly head at **0.969 exact /
0.024 head gap**, against the ZINC-derived head's 0.966. In 73 min, not 158.

ZINC's full account: it costs −0.0125 H@1 on retrieval, its frozen encoder is
far worse on tastepocket (0.2328 vs 0.3250), it oversmooths more (41% vs 33%),
and its last remaining justification — the assembly head — is now matched by
flavour-only training. **The stage can be retired.**

## 6. Pocket conditioning, retested at width 512: worse

The earlier null was measured on a capacity-starved encoder, so it was worth
re-testing once that changed. It did change — in the wrong direction.

| | pocket-trained | no-finetune | Δ |
|---|---:|---:|---:|
| width 300 | 0.1622 | 0.1947 | −0.0325 |
| **width 512** | 0.1119 | 0.1882 | **−0.0763 (−4.9 SE)** |

Worse in 5 of 5 folds, and the harm more than doubles with the better encoder.
A wider encoder produces a better flavour representation, and finetuning it on
**243 records** damages a good representation more than a weak one. More
capacity means more to lose.

## What this changes

- **Use width 512, lr 1e-4, τ 0.01, no freezing.**
- **Retire the ZINC stage.**
- **Do not finetune on tastepocket**, and do not condition on pockets.
- **Re-read every pre-sweep conclusion as provisional.** They were all measured
  at width 300 and lr 1e-3. Two have already changed: freezing reversed sign,
  and pocket conditioning went from null to actively harmful.

## Method notes

Two bugs in the sweep harness, both caught before they produced numbers, both
worth recording because they are the same failure in different clothes.

**Arms trained at `condvec_dim=0`.** The training command omitted it and
inherited the config default; the baseline uses 24. Four arms trained
flavour-blind, returned rc=0, wrote checkpoints and logged 218 metrics each.
Only a scoring shape-mismatch exposed it — and only because scoring builds the
model from a declared config rather than inferring it from the checkpoint. Fix:
shape-affecting overrides live in ONE list read by both training and scoring,
plus `assert_trained_as_scored()`, which reads the resolved Hydra config and
refuses to score an arm that trained differently.

**The guard then produced a false positive.** It matched on the leaf name
`temperature`, and four blocks define one (graph 0.1, linker 0.05, rgroup 0.01,
assembly 0.1). It compared the wrong one and failed all eight Stage B arms
whose training was correct. Now resolved via `OmegaConf.select` on the full
dotted path. It cost a scoring pass; the opposite error would have cost the
stage.

## Reproducing

```bash
cd molplatte/src
python3 scripts/run_sweep.py --stage A     # s1 variants
python3 scripts/run_sweep.py --stage W     # width scan
python3 scripts/run_sweep.py --stage B1    # tau x lr x freeze
python3 scripts/run_sweep.py --stage BLR   # lr extension
python3 scripts/run_sweep.py --stage D     # seed replication
```

Results append to `~/checkpoints/molplatte/sweep_results.csv`. Resumable: an arm
whose log exists is skipped.

## Limitation

Every s2 arm warm-starts from the same single-seed `sw-s1-wide512`, so the seed
statistics in §4 bound s2 variance, NOT end-to-end pipeline variance. If s1
variance is comparable the real intervals are wider. Re-seeding s1 is ~1.6 h per
seed and remains undone.
