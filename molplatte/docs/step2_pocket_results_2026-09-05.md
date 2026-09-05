# STEP 2 — pocket-conditioned finetuning: results

Run 2026-09-05. Seed 911012, 40 epochs per fold, GPU1.
Corpus `tastepocket_corpus` (243 records / 734 decompositions / 906 R-groups),
scored against the 91,935-row union library (effective size 946).

## The headline

**Neither the pocket half nor the finetuning stage does anything.**

- vs an exact pocket-off control: mean delta H@1 **+0.0008**
- vs no finetuning at all (STEP 1 checkpoint evaluated directly): **-0.0106**,
  worse in 3 of 5 folds

The pocket is not being ignored -- it contributes 48% as much as flavour to the
query vector and varies across receptors. It is present, substantial, and
carries nothing retrieval can use. The best available model for tastepocket
retrieval is the STEP 1 flavour-only checkpoint, untouched.

## STEP 1 (ligand-only pretraining)

`coconut-flavordb-full`, 30 epochs, condvec 24 flavour bits.

| | value | prior | lift |
|---|---:|---:|---:|
| H@1 | 0.3235 | 0.0249 | **13.0x** |
| H@10 | 0.5665 | 0.0905 | 6.3x |
| MRR | 0.4277 peak | | |

## STEP 2 (five folds)

| fold | N | MRR | H@1 | H@10 | H@100 |
|---|---:|---:|---:|---:|---:|
| 0 | 178 | 0.2318 | 0.1180 | 0.4494 | 0.7360 |
| 1 * | 197 | 0.0991 | 0.0558 | 0.2183 | 0.3401 |
| 2 | 141 | 0.2164 | 0.1631 | 0.2908 | 0.4113 |
| 3 | 142 | 0.1936 | 0.1197 | 0.3169 | 0.4859 |
| 4 | 138 | 0.3418 | 0.2536 | 0.5000 | 0.6667 |

\* fold 1 holds out TRPV1+TRPA1 entirely (52 of 58 records). Both are single
components of the (ligand, receptor) graph, so this fold tests an unseen receptor
FAMILY -- a harder question than the others ask. It is the worst fold by a wide
margin, which is the expected direction.

Unseen-receptor folds (0, 2, 3, 4): **H@1 0.1636 +- 0.0635**, prior 0.0293,
lift 5.6x. The spread is large because a fold is ~50 records.

## The ablation, obtained by accident

The first sweep ran with a bug that left the pocket path disconnected
(|pocket output weight| = 0.0000 in every fold). That makes it an exact
pocket-off control: same seed, same folds, same data, same epochs.

| fold | H@1 pocket-on | H@1 pocket-off | delta |
|---|---:|---:|---:|
| 0 | 0.1180 | 0.1180 | +0.0000 |
| 1 | 0.0558 | 0.0558 | +0.0000 |
| 2 | 0.1631 | 0.1631 | +0.0000 |
| 3 | 0.1197 | 0.1156 | +0.0041 |
| 4 | 0.2536 | 0.2536 | +0.0000 |

**mean delta H@1 +0.0008, H@10 +0.0021.**

## The pocket is not being ignored

That matters, because "no effect" has two very different causes and this
distinguishes them. Measured on the final checkpoint over all 906 condvec rows:

```
projected pocket half     |x| mean 1.1495   (flavour half 0.0618)
contribution to query     pocket 6.49  vs  flavour 18.69   -> ratio 0.477
distinct pocket vectors   241 of 906 rows
per-dimension std         1.225
```

So the model learned to inject a substantial, receptor-varying signal into the
query -- roughly half the magnitude of the flavour signal -- and retrieval did
not move. The pocket representation is present and useless, not absent.

## Novel retrieval

| | base | novel |
|---|---:|---:|
| H@1 | 0.1561 | 0.0000 |
| H@10 | 0.3869 | 0.0000 |
| H@100 | 0.5733 | 0.0333 |

Across all folds, 76 novel targets, 1 retrieved at K=100. Base-rate expectation
if novel retrieval matched base would be ~44. Novel R-groups have median corpus
count 1 against 255 for base targets, so this is retrieval of familiar chemistry
only.

Per fold the novel N is small (2 to 27), so no single fold's 0.0000 means much;
the aggregate is what carries the claim.

## What this does and does not license

Supported:
- The pocket half as built (frozen ESM-2, mean-pooled 10 A residues, learned
  linear reduction) adds nothing to R-group retrieval on 269 records.
- The model does not generalise to R-groups absent from pretraining.

NOT supported:
- That pocket conditioning cannot work. 269 records is far too few to learn a
  pocket->chemistry mapping, and the pooled embedding identifies the receptor
  family with 98.5% accuracy, so the easiest thing for the model to learn is
  receptor identity -- which is useless on a held-out receptor by construction.
- Any claim about 3D geometry. Nothing here tested it.

## The second control: does finetuning help at all?

The expanded STEP 1 checkpoint, evaluated on each fold's held-out set with no
training. It is numerically the flavour-only STEP 1 model, because
PocketConditioning's output layer is zero-initialised in a fresh model, so the
projected pocket is the zero vector.

| fold | no-finetune H@1 | finetuned H@1 | delta | no-ft MRR | ft MRR |
|---|---:|---:|---:|---:|---:|
| 0 | 0.2022 | 0.1180 | **-0.0842** | 0.2802 | 0.2318 |
| 1 * | 0.0718 | 0.0558 | -0.0160 | 0.1386 | 0.0991 |
| 2 | 0.1135 | 0.1631 | +0.0496 | 0.1721 | 0.2164 |
| 3 | 0.1348 | 0.1197 | -0.0151 | 0.2112 | 0.1936 |
| 4 | 0.2411 | 0.2536 | +0.0125 | 0.2983 | 0.3418 |

**mean delta H@1 -0.0106, H@10 +0.0104, MRR -0.0035. Worse in 3 of 5 folds.**

Per-fold deltas swing from -0.084 to +0.050 with no consistent sign, which is
what noise looks like at ~50 records per fold. Finetuning on tastepocket buys
nothing measurable and may cost a little.

Minor caveat: the control runs through the `test` dataloader and the STEP 2
numbers are the last `val` epoch. Both are the same held-out fold -- `_fold_splits`
returns it for both -- but the query counts differ slightly (183 vs 178 on fold
0, ~3%). Far smaller than the deltas above, so it does not change the reading.

## So what is worth doing next

1. **CrossDocked pocket embeddings.** 11,268 ligands over 147,648 pairs, against
   269 records here. It is the only substrate with enough receptor variety to
   learn a pocket-to-chemistry mapping rather than receptor identity. Full
   receptors are at `crossdocked_v1.1_rmsd1.0/<target>/*_rec.pdb`, 2,474 target
   directories; use those, not the pre-cut `*_pocket10.pdb`.
2. **A pocket shuffle test.** Permute the pocket half across records, exactly as
   the flavour shuffle test does. Given the pocket contributes 48% of flavour's
   magnitude while changing nothing, the prediction is `cond - shuf` near zero --
   and confirming that would establish the signal is inert rather than merely
   redundant with the core.
3. The EGNN ablation is NOT next. Geometry cannot be the bottleneck while the
   sequence-derived signal is already being injected at half the flavour
   magnitude and doing nothing.
