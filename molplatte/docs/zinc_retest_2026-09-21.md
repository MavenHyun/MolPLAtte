# ZINC pretraining retested at width 512 on a clean split: the verdict holds, five times larger

**ZINC2020 pretraining costs -0.0653 H@1 (-19% relative).** The September
verdict (-0.0125) was taken at hidden_dim=300 / lr=1e-3, six days before the
sweep that showed width and learning rate were the two untuned parameters worth
+19% -- the same regime in which encoder freezing "helped" and then reversed
sign. Every objection to that measurement has now been addressed and the answer
got STRONGER.

## Result

| arm | step 1 | step 2 (final) | H@10 | MRR | novel@10 |
|---|---:|---:|---:|---:|---:|
| **F — flavour only** | 0.3220 | **0.3446** | **0.6627** | **0.4513** | 0.0000 |
| **Z — ZINC2020 + flavour** | 0.1530 | **0.2793** | 0.5955 | 0.3849 | **0.0417** |
| Δ (Z − F) | | **-0.0653** | -0.0672 | -0.0664 | +0.0417 |

One scoring path for all four: library 86,776, effective 944, N=20,000,
prior@1 0.0243.

**Step 2 cannot recover the gap.** ZINC step 1 reached 0.1530 on flavour
retrieval after ~200k optimiser steps; 20 epochs of flavour finetuning lifted it
to 0.2793, still far short of the 0.3446 flavour-only reached FROM SCRATCH. The
ZINC representation is worse as a starting point than random init plus flavour
training -- not merely slower to converge.

**The one place ZINC wins** is `novel@10` (0.0417 vs 0.0000): it does retrieve
some fragments outside the flavour base. That does not offset -0.065 H@1.

**Cost:** 236.6 min vs 92.1 min for step 1 -- 2.6x the compute for a
substantially worse model.

## What made this a fair test

Three defects in the earlier comparison were fixed first:

1. **Molecule-level splitting.** Steps 1-2 had split by DECOMPOSITION (2.91 cuts
   per molecule), so 91.9% of eval items came from molecules also in training.
   `split_by=molecule` assigns whole molecules by a stable hash of
   (seed, mol_id) -- independent of corpus size, so a molecule lands in the same
   split in EVERY corpus containing it. Straddling molecules 93,226 -> 0.
2. **Train-only R-group library.** Each entry's `count` feeds log p(k), half the
   scoring function, so counting evaluation molecules put the test set into the
   prior. Rebuilt from 353,814 training molecules: 86,776 entries (eff 944) vs
   91,941 (eff 946).
3. **No structural overlap.** ZINC ids and COCONUT ids are different namespaces,
   so duplicates were found on WASHED CANONICAL SMILES -- the builder's own
   dedup key. 16,806 ZINC duplicates of flavour structures dropped, plus 130
   tastepocket structures reserved for step 3.

Merged step-1 corpus: **10,236,427 molecules / 34,642,267 decompositions**
(9,843,424 ZINC-only + 393,003 flavour).

## Epochs were matched on OPTIMISER STEPS, not passes

ZINC ran 3 epochs over 34.6M decompositions (~200k steps); flavour-only ran 30
epochs over 1.15M (~67k steps). ZINC received **3x more gradient steps**, so
under-training is not an available explanation.

## Caveats

- **Single run per arm.** The -0.0653 is ~50x the seed SE seen in earlier sweeps
  (~0.0013), so noise is very unlikely, but this is not replicated.
- Numbers are NOT comparable to the historical 0.3701 / 0.3648: the split, the
  library and the prior all changed. Only arm Z vs arm F is a clean comparison,
  which is why the flavour arm was retrained rather than reused.
- `novel@10` is measured relative to the BASE vocabulary. With a flavour-derived
  base, flavour targets are in-base by construction, which is why arm F reads
  0.0000. It is not comparable across vocabulary choices.

## Reproduce

```bash
python molplatte_preprocess/scripts/merge_pretrain_corpus.py
python molplatte_preprocess/enumerate_rgroups.py --corpus <merged> --ids-file <train ids>
python molplatte/src/scripts/run_zinc_chain.py --gpu 1 --arms zinc,flavour
```
