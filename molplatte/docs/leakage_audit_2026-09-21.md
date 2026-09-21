# Leakage audit: two paths, found while preparing the ZINC retest

Checking the overlap requirement for a combined ZINC+flavour step-1 corpus
surfaced two pre-existing leaks. Neither invalidates a RELATIVE conclusion --
every arm in every sweep shared the same split, and the pocket null is a
real-versus-permuted comparison that leakage affects symmetrically. Both inflate
ABSOLUTE numbers, including the shipped model's headline.

## 1. Steps 1-2 are split by decomposition, not by molecule

`random_split` partitions the corpus at the DECOMPOSITION level, and a molecule
carries **2.91** decompositions on average. A molecule with four cuts routinely
lands three in train and one in test.

| | count |
|---|---:|
| molecules with >=1 item in TRAIN | 384,692 |
| molecules with >=1 item in VAL/TEST | 101,600 |
| molecules in **BOTH** | **93,226** |
| eval molecules never seen in training | **8,374** (8.2% of eval molecules) |
| **eval ITEMS whose molecule was also trained on** | **105,338 / 114,574 (91.9%)** |

The model is tested on a different cut of a molecule it trained on -- same
scaffold, same R-group vocabulary, frequently overlapping fragments.

**Affected:** every step-1/step-2 number, including H@1 0.3098 -> 0.3701 (+19%)
and novel@10 +25%, the hyperparameter sweep, `FINDINGS.md` and the published
artifact.

**Not affected:** step 3. The pocket CV holds out whole connected components of
the (ligand, receptor) graph, so molecules and receptors are disjoint by
construction. The pocket work was measured on a CLEAN split while steps 1-2 were
measured on a leaky one -- which makes any direct comparison between their
absolute numbers unsound.

## 2. Tastepocket ligands appear in the step-2 training corpus

| | count |
|---|---:|
| distinct tastepocket structures | 130 |
| also present in `coconut-flavordb-full` | **63 (48%)** |
| of those, in step-2's TRAIN split | **44** |
| of those, held out in step 2 | 19 |

So **44 of 130 (34%)** of tastepocket compounds were trained on during step 2,
without their pockets. When step 3 evaluates on a held-out receptor, the model
has already seen a third of the ligand chemistry.

This is a plausible partial explanation for the no-finetune baseline performing
as well as it does on unseen receptors, and it means the pocket had even less
left to contribute than the conditioning experiments assumed.

## 3. ZINC overlap (the question that started this)

| | count |
|---|---:|
| ZINC ∩ flavour | 16,804 structures (0.17% of ZINC, 4.3% of flavour) |
| ZINC ∩ tastepocket | 2 structures |

Small. A combined step-1 corpus is genuinely ~10.25M distinct structures, so the
ZINC retest is a real experiment rather than a disguised duplicate. Molecule ids
cannot detect this -- `ZINC000000725873` and `CNP0005390` can be the same
structure -- so the comparison is on WASHED CANONICAL SMILES, the same key the
corpus builder dedups on.

## Why a combined step-1 corpus needs explicit handling

`random_split` partitions INDEX POSITIONS with `manual_seed(42)`, so the
partition depends on corpus SIZE. Steps 1 and 2 currently avoid cross-stage
leakage only by accident: they use the same corpus, so they draw the same split.
A combined step-1 corpus is a different size and would therefore train on
step-2's test molecules.

Any combined-corpus chain must exclude step-2's val+test molecules from step 1
explicitly (101,600 molecules, 25.8% of the flavour corpus).

## Recommended fix

1. Split by MOLECULE, not decomposition -- every cut of a molecule on one side.
2. Exclude tastepocket structures from step-1/2 training.
3. Exclude step-2's held-out molecules from any combined step-1 corpus.

Costs one retrain of steps 1-2. Expect the reported H@1 to FALL, possibly a lot:
only 8.2% of the current eval set is genuinely unseen.
