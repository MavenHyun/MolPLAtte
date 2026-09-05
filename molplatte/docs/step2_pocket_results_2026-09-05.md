# STEP 2 — pocket-conditioned finetuning: results

Run 2026-09-05. Seed 911012, 40 epochs per fold, GPU1.
Corpus `tastepocket_corpus` (243 records / 734 decompositions / 906 R-groups),
scored against the 91,935-row union library (effective size 946).

## The headline

**The pocket half is used and does not help.** It contributes 48% as much as the
flavour half to the query vector, varies across receptors, and changes retrieval
by +0.0008 H@1.

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

## Missing control

There is no "STEP 1 checkpoint evaluated on the folds without finetuning" number,
so the gain from finetuning at all is unmeasured, separate from the gain from
the pocket. That is the cheapest next measurement.
