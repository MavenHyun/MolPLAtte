# ARCHIVED — results from the macfrag / condition-vector era

> **Superseded. Do not cite these numbers.** Archived 2026-08-20.
>
> Every result below was measured on a corpus or configuration that no longer
> exists. Kept because the *reasoning* is still load-bearing — several of these
> measurements are why the current design looks the way it does — but the
> figures describe a different system.
>
> What changed underneath them:
>
> | | then | now |
> |---|---|---|
> | decomposition | `macfrag` re-framed through the fragment tree | `naveja_recap`, ratio 0.5 |
> | condition vector | 97-dim R-group functional groups | **removed** — it leaked the answer into the query |
> | WL hash | v1 (`chiral_tag`, `bond_dir`, `is_conjugated` in the label) | v2 |
> | retrieval scoring | pure similarity | + `log p` popularity correction |
> | charge | neutralised by `wash()` | preserved |
> | ring bonds | never cut | cut outside small rings |
> | sampling | one core per molecule per epoch | every (molecule, core) pair |
>
> The three findings worth carrying forward, all still true:
>
> 1. **`naveja_recap` yields ~1 R-group per decomposition** — measured at 1.04
>    on a 400-molecule probe, 1.07 and 1.08 at corpus scale. This is why the
>    fragment-tree adapter exists, and it is the trade-off accepted in choosing
>    naveja for its far less degenerate library.
> 2. **The condition vector was leakage.** Zeroing it at inference took hit@1
>    from 0.6111 to exactly 0.0000 on the queries that carried one.
> 3. **Pure-similarity scoring understates a PMI-trained retriever.** Adding
>    `log p` back moved lift from 0.61x to 2.20x, reproducing MolDAM devlog
>    Phase 9 -> Phase 10 independently.
>
> Current corpus statistics: `molplatte_corpus_eda.pdf`.

---

## 5. Measured results

### 5.1 Decomposition on flavor chemistry

400 FlavorDB molecules, heavy atoms in [5, 50], median 23:

| method | no-decomp | dec/mol | R-groups/dec | k >= 2 | throughput |
|---|---|---|---|---|---|
| `naveja_recap` (ratio 2/3) | 0.2% | 10.65 | **1.04** | 3.70% | 678 mol/s |
| `naveja_recap` (ratio 1/2) | 0.2% | 12.81 | 1.06 | 5.54% | 634 mol/s |
| `naveja_recap` (ratio 1/3) | 0.2% | 13.67 | 1.10 | 8.34% | 598 mol/s |
| `bemis_murcko` | 18.2% | 1.00 | 4.94 | 77.4% | 9749 mol/s |
| `macfrag` (raw partition) | 12.2% | 1.00 | 7.52 frags | — | 235 mol/s |
| `synton` (raw partition) | 50.2% | 1.00 | 3.36 frags | — | 7.2 mol/s |

**The headline finding.** `naveja_recap` — MolPLA's own decomposition method —
yields ~1 R-group per decomposition on flavor chemistry. MolPLA's `islinked`
subset space is `2^k - 1`, so at `k = 1` there is exactly one instance per core
and the core-decoration objective has nothing to decorate. This reproduces the
same structural failure MolDAM_prep measured on ZINC (99.42% single-R-group at
ratio 2/3) and confirms the cause is the decomposer, not the chemistry: RECAP
children are single connected fragments, so `M − core` is one pendant group
regardless of the size threshold.

### 5.2 The fix — re-framing multi-cut partitions as anchored stars

A partition induces a **fragment tree** (vertices = fragments, edges = cut bonds;
acyclic because ring bonds are never cut). Any *connected* vertex subset taken as
the core leaves one component per boundary edge, each attached by exactly one cut
bond — a genuine star. Enumerating connected subsets above `ratio * n_atoms`
reproduces MolPLA's "single molecule, multiple putative cores" property.
Implemented in `molplatte_prep/anchored_from_partition.py`.

Same 400 molecules, `macfrag` re-framed:

| ratio | no-decomp | cores/mol | R-groups/core | k >= 2 | core nHA |
|---|---|---|---|---|---|
| 0.33 | 12.2% | 79.30 | 3.26 | 89.7% | 20.7 |
| **0.50** | 12.2% | 64.45 | **3.34** | **89.7%** | **22.5** |
| 0.67 | 16.8% | 45.65 | 3.23 | 88.6% | 24.7 |

MolPLA reports ~4.04 cores per molecule and ~20.8-heavy-atom cores on GEOM, so
`ratio=0.5` matches its core size; `--max-cores 10` prunes the surplus (MolPLA
itself dropped molecules above 10 cores, its 99th percentile being 11).

### 5.3 Corpora built

FlavorDB: 25,595 compounds, 23,379 after the `[5, 50]` heavy-atom filter.

| corpus | records | dec/mol | R-groups/dec | size |
|---|---|---|---|---|
| `flavordb_full/macfrag` | 20,428 | 7.27 | **2.69** | 242 MB |
| `flavordb_full/naveja_recap` | 23,216 | 8.76 | **1.07** | 231 MB |
| `flavordb_full/bemis_murcko` | 19,121 | 1.00 | 5.17 | 166 MB |
| `flavordb_full/synton` | 12,084 | 2.52 | 1.12 | 111 MB |

The corpus-scale `naveja_recap` number (1.07 R-groups/decomposition over all
23,216 records) confirms the 400-molecule probe. `synton` retained only 48.3% of
molecules, matching its measured 50.2% no-partition rate.

COCONUT: 737,343 records collapsing to 489,395 distinct compounds after variant
deduplication, 396,936 after the heavy-atom filter.

| corpus | records | dec/mol | R-groups/dec | size |
|---|---|---|---|---|
| `coconut_full/macfrag` | 354,515 | 7.56 | **3.13** | 4.1 GB |

Natural products are larger and more decorated than flavor volatiles, so they give
*more* multi-R-group structure than FlavorDB (3.13 vs 2.69 R-groups per
decomposition) — which is the property MolPLA's objectives need. Build throughput
was 2,620 mol/s on 88 workers; `no_decomp` was 10.7%.

### 5.4 Integration validation

A 16-molecule batch off `flavordb_full/macfrag` through the full stack:

```
W               : 64 graphs / 918 nodes   (16 G + 16 P + 32 R)
joints (J)      : 32          samples (B) : 16
R_hashes        : 32 keys, 17 distinct    dup_rate : 0.69
losses          : graph 0.975  linker 3.888  rgroup 10.791
gradient flow   : 88/88 parameter tensors received non-zero gradient
```

**`dup_rate = 0.69`** — 69% of retrieval queries in a batch have a chemically
identical counterpart elsewhere in that batch. Plain InfoNCE would treat those as
negatives and actively push identical structures apart. This is what the
multi-positive masking in `loss_modules/contrastive.py` exists for, and the
number justifies it empirically rather than by analogy to MolDAM.

---

## 5.5 The R-group library (the RGR task)

MolPLA's R-Group Retrieval framework does **not** score against in-batch
negatives. It embeds *every recommendable R-group in the corpus* — 61,279 of them
on GEOM — with the current R-group projector, indexes them with FAISS, and
retrieves the top 1000 per query. MRR and Hit@{5,10,20,50,100,500,1000} are
computed over that library. Without it there is no lead-optimization task; an
in-batch gallery measures a much easier problem under a similar name.

MolPLAtte builds it in two halves, because they have different lifetimes:

| half | built by | depends on | rebuilt |
|---|---|---|---|
| **vocabulary** — distinct R-groups, canonical masked graphs, counts, condvecs | `molplatte_preprocess/enumerate_rgroups.py` | the corpus only | once per corpus |
| **vector library** — those graphs embedded + FAISS index | `callbacks/RGroupLibraryRetrieval.py` (training) / `molplatte/src/build_library.py` (inference) | the projector's current weights | **every validation epoch** |

The second half must be rebuilt continuously: a library embedded at epoch 3 is
meaningless for a query embedded at epoch 7. MolPLA rebuilds it at every
validation; so does the callback.

**Keying.** MolPLA keys the vocabulary on the R-group's masked SMILES. MolPLAtte
keys on the **WL subgraph hash** and carries SMILES as a label, because a masked
linker atom is not a real chemical entity — two structurally different masked
graphs can strip to the same SMILES (`*O` appears twice in the FlavorDB
vocabulary with different counts). The hash is already what the corpus stores per
R-group and what the contrastive loss uses for multi-positive grouping, so keying
on it keeps one identity notion throughout.

### Vocabularies built

| corpus | distinct R-groups | occurrences | **effective size** | top-1 share |
|---|---|---|---|---|
| `flavordb_full/macfrag` | 5,663 | 399,141 | **31** | 19.31% |

The most frequent entries are chemically sensible: `*O`, `*CO`, `*C`,
`*c1ccccc1`, `*c1ccc(O)c(O)c1` (catechol), and glycoside sugars — the last
unsurprising given FlavorDB is sugar-heavy.

### Why every Hit@K is logged next to a prior baseline

**Effective size 31 out of 5,663 distinct entries.** One R-group is 19% of all
occurrences. This is not a 5,663-way retrieval problem, and Hit@K against it is
mostly a measurement of the frequency prior. The callback therefore logs three
numbers per cut-off:

```
library/hit@K        the model
library/prior_hit@K  the constant "always return the K most frequent" predictor
library/lift@K       the ratio
```

Measured on an **untrained** model over the full 5,663-row library:

| K | hit@K | prior hit@K | lift |
|---|---|---|---|
| 1 | 0.0000 | 0.156 | 0.00 |
| 10 | 0.0053 | 0.640 | 0.01 |
| 100 | 0.0079 | 0.857 | 0.01 |
| 1000 | 0.0979 | 0.947 | 0.10 |

An untrained model reporting "Hit@1000 = 0.098" looks non-trivial until you see
that always returning the 1000 most common R-groups scores 0.947. This is the
mistake MolDAM made — its headline figure was framed as "47× above random", and
re-scored against the frequency prior it landed *at* the prior. Random is the
wrong reference. **`lift <= 1` means the model has learned nothing the prior does
not already give you, whatever `hit@K` says.**

Both retrieval evaluations are logged, under distinct prefixes so they cannot be
confused: `{stage}/faiss/*` is the cheap val-split gallery, `{stage}/library/*`
is the full corpus library.

### Does it learn? — 25 epochs on `flavordb_full/macfrag`

Full-library retrieval over all 5,663 R-groups, ~1,450 validation queries per
epoch, batch 512, bf16, one GPU:

| metric | untrained | epoch 7 | **epoch 25** | prior | **lift** |
|---|---|---|---|---|---|
| MRR | 0.0016 | 0.497 | **0.723** | — | — |
| Hit@1 | 0.000 | 0.374 | **0.645** | 0.140 | **4.6x** |
| Hit@10 | 0.005 | 0.675 | **0.847** | 0.650 | **1.30x** |
| Hit@100 | 0.008 | 0.798 | **0.952** | 0.823 | **1.16x** |

The trajectory is the informative part. At epoch 7 the model beat the prior at
Hit@1 (2.5x) but sat *at or below* it at Hit@10 (1.04x) and Hit@100 (0.97x) — it
had learned to rank the top of the list without beating "return the most common"
in the tail. That is exactly the regime MolDAM's headline number was reported
from, and reporting Hit@100 = 0.798 at that point without the prior beside it
would have been misleading. By epoch 25 it clears the prior at every cut-off.

### Qualitative check — is it chemistry or is it memorisation?

Hit@K says the right row comes back; it does not say the *wrong* rows are
sensible. Querying the exported library with core templates from held-out
molecules (top-5 shown, true R-group in bold where retrieved):

```
true  *OP(=O)(O)OP(=O)(O)OP(=O)(O)O          (triphosphate)
top-5 *OP(=O)(O)O | *OP(=O)(O)OP(=O)(O)O | ***OP(=O)(O)OP(=O)(O)OP(=O)(O)O** | ...
      -> mono-, di-, tri-phosphate in order: the homologous series

true  *NCCCC                                  (n-butylamine)
top-5 *NCCC | *NCCCCN | *NCCCN | *NCCN | *NCC
      -> the alkylamine homologous series

true  *C1O[C@H](C(=O)O)[C@@H](O)[C@H](O)[C@H]1O   (glucuronic acid)
top-5 exact match at rank 1, then its stereoisomers

true  *CCCCCCCCC                              (n-nonyl)
top-5 *CCC | *CC | *COC | *C | *CCCC
      -> right family, wrong chain length. An honest miss.
```

The failure mode is informative: the model has learned R-group *families* — it
never answers a phosphate query with an alkylamine — but it does not resolve
chain length within a family. That is what MolPLA claims qualitatively
("rationally suggesting R-group replacements"), reproduced here on flavor
chemistry, and it is what a lead-optimization user would actually want from a
first pass.

### Combined corpus — the primary run

`coconut-flavordb_full/macfrag`: 369,881 molecules (351,382 COCONUT + 18,499 FlavorDB,
5,062 duplicates removed), 332,893 train / 18,494 val / 18,494 test.
5.4M parameters, 30 epochs, batch 512, bf16, one GPU. Library: **60,773 distinct
R-groups**, effective size 34, top-1 share 31.08% — essentially the same scale as
MolPLA's 61,279 on GEOM, so these numbers are comparable in a way the
FlavorDB-only ones were not.

| epoch | MRR | Hit@1 | Hit@10 | Hit@100 |
|---|---|---|---|---|
| 1 | 0.625 | 0.535 | 0.768 | 0.873 |
| 8 | 0.778 | 0.698 | 0.912 | 0.979 |
| 16 | 0.796 | 0.720 | 0.922 | 0.982 |
| **30** | **0.811** | **0.741** | **0.933** | **0.987** |
| *prior* | — | *0.265* | *0.671* | *0.809* |
| **lift** | — | **2.80x** | **1.39x** | **1.22x** |

val/loss 10.09 -> 5.79. Retrieval clears the frequency prior at every cut-off from
epoch 1 onward — unlike the FlavorDB-only run, which needed ~25 epochs to beat the
prior in the tail. The larger, less degenerate library is the likely reason: with
60,773 rows at effective size 34 there is more genuine structure to learn than in
5,663 rows at effective size 31.

Queries are subsampled to 20,000 of ~30,000 per epoch with a fixed seed
(`library/queries_truncated` records the rest).

### Hash v1 vs v2 — what the fix was actually worth

Identical settings, identical corpus content; the only difference is the WL hash
(see §6). 30 epochs, macfrag. The two corpora are
`coconut-flavordb_full_hashv1` (superseded, kept for this comparison) and
`coconut-flavordb_full` (canonical).

| | library | MRR | Hit@1 | lift | Hit@10 | lift | Hit@100 | lift |
|---|---|---|---|---|---|---|---|---|
| v1 (old hash) | 60,773 | 0.8114 | 0.7409 | 2.80x | 0.9326 | 1.39x | 0.9866 | 1.22x |
| **v2 (fixed)** | **48,671** | **0.8264** | **0.7534** | 2.82x | **0.9470** | 1.41x | **0.9924** | 1.22x |
| delta | -19.9% | +0.0150 | +0.0125 | | +0.0144 | | +0.0058 | |

**The fix was worth less than predicted.** I had said the split-row artefact meant
the v1 numbers "understate the true figure". Directionally right -- every metric
improves -- but by ~1.3 points of Hit@1, and the **lift is essentially unchanged**
(2.80x -> 2.82x) because the frequency prior barely moved (0.2646 -> 0.2676).
Merging duplicate rows removed mass from the numerator and denominator alike.

Two things also confound the raw comparison and should keep it from being read as
a clean win:

1. The v2 library is **19.9% smaller**, which makes retrieval intrinsically
   easier. Some of the +0.0125 is a smaller haystack, not better measurement.
2. `val/loss` moved the *wrong* way (5.793 -> 5.930) and is **not comparable
   across versions**: fewer distinct hashes means more in-batch multi-positive
   collisions, so the per-row loss is averaged over more positives and the scale
   shifts.

The fix is still correct -- a vocabulary key must be isomorphism-invariant, and
`chiral_tag`/`bond_dir` are not -- but its value is correctness of the artefact,
not a materially better score. Convergence also differs: v2 starts *lower*
(Hit@1 0.383 vs 0.535 at epoch 1) and overtakes by epoch 16, consistent with the
changed multi-positive structure altering the early gradient signal.

### Measured cost of the all-zero condition vectors

64.9% of condition vectors are all-zero (see §6). Splitting validation queries by
whether the target R-group is a pure alkyl chain — the case where the condition
carries **no** information at all — quantifies what that costs:

| query group | Hit@1 | Hit@10 |
|---|---|---|
| pure-alkyl R-groups (all-zero condition) | **0.398** | 0.958 |
| all other R-groups | **0.692** | 0.934 |

Hit@1 drops 42% relative, while Hit@10 is *higher*. The model finds the right
R-group **family** reliably and fails on **size**: 37 of 191 top-1 errors are
alkyl-to-wrong-length-alkyl (`*CCC`->`*CC`, `*CCCCCC`->`*CCCC`). That is exactly
the signal the condition vector cannot express, and it is milder than predicted
only because the core context partially substitutes for it.

**Not comparable to MolPLA's published numbers** (MRR 0.2616, R@10 0.4839,
R@100 0.8702 on GEOM): that library has 61,279 distinct R-groups against this
one's 5,663, with a far less degenerate frequency distribution. A higher MRR here
reflects an easier library, not a better model.

---

## 5.6 Assembly head — 30 epochs, head on vs off

Identical settings, `assembly.enabled` the only difference. Both logged to wandb
project `NoahsFarm_MolPLAtte`.

| metric | baseline | assembly | delta |
|---|---|---|---|
| library MRR | 0.8147 | 0.8285 | +0.0138 |
| library Hit@1 | 0.7394 | 0.7546 | +0.0152 |
| library Hit@10 | 0.9420 | 0.9502 | +0.0081 |
| Hit@1 lift over prior | 2.78x | 2.84x | |
| val/loss | 5.685 | 5.467 | -0.218 |
| val/loss graph | 0.7403 | 0.6007 | -0.140 |
| val/loss linker | 6.5209 | 6.3825 | -0.138 |
| val/loss rgroup | 4.2930 | 4.2209 | -0.072 |

**Retrieval did not degrade.** That was the stated risk -- MolPLA names
"adversarial optimization trajectories incurred by three different loss
objectives" as its unresolved limitation, and this adds a fourth. Instead all
three contrastive losses *improved*, which is what an auxiliary objective that
regularises the shared encoder looks like rather than one that competes with it.

### But the effect is inside the noise floor

Two **baseline** runs, same seed, same 30 epochs, differing only in
`num_workers` (which reorders dataloader draws and so changes which
`(decomposition, islinked)` pairs are sampled):

```
num_workers=24   Hit@1 = 0.7534
num_workers=12   Hit@1 = 0.7394
spread                  0.0140
assembly delta          0.0152
```

The improvement is the same size as baseline-to-baseline variance. **It should
not be reported as an effect** without seed replicates -- 3-5 per arm would
establish whether it survives. The loss improvements are more consistent across
terms and somewhat more persuasive, but they are also a single sample.

### Recovery accuracy needs its own baseline

`recover_acc = 0.9948` looks conclusive and is not. Measured majority-class
rates on the corpus:

| target | classes | majority | model | headroom closed |
|---|---|---|---|---|
| `hybridization` | 3 | 0.564 | 0.9922 | **98.2%** |
| `is_conjugated` | 2 | 0.653 | 0.9965 | **99.0%** |
| `atomic_num` | 4 | 0.715 | 0.9813 | **93.4%** |
| `is_aromatic` | 2 | 0.831 | 0.9998 | 99.9% |
| `num_explicit_hs` | 2 | 0.862 | 0.9938 | 95.5% |
| `bond_type` | 3 | 0.873 | 0.9949 | 96.0% |
| `bond_stereo` | 3 | 0.989 | 0.9950 | 55.4% |
| `formal_charge` | **1** | 1.000 | 0.9997 | degenerate |
| `edge_is_aromatic` | **1** | 1.000 | 1.0000 | degenerate |

Two targets are **structurally constant on this corpus** and I should have
predicted both: `wash()` neutralises charges, so a joint atom is always neutral;
and cut bonds are never ring bonds, so the reformed bond is never aromatic. They
scored ~100% for free and padded the macro average.

Dropped from the default target set. The loss now logs `majority` and
`headroom` beside every attribute and excludes degenerate targets from the macro
mean -- the same discipline as `library/lift@K`. Real learning is on
`hybridization`, `is_conjugated` and `atomic_num`, where the model closes
93-99% of a genuine gap.

---

## 5.7 Seed sweep — does the assembly head help? (5 seeds x 2 arms)

Condvec removed, training-side logQ off, 30 epochs, `coconut-flavordb_full/macfrag`.
`num_workers` pinned at 12 across every run, and `data_module_kwargs.seed` swept
alongside `random_seed` so the replicates capture data-sampling variance and not
just weight init.

| metric | baseline (n=5) | assembly (n=5) | delta | paired t |
|---|---|---|---|---|
| corrected hit@1 | 0.5872 +- 0.0149 | 0.5862 +- 0.0164 | -0.0011 | p=0.92 |
| corrected hit@10 | 0.7441 +- 0.0245 | 0.7408 +- 0.0261 | -0.0033 | |
| corrected MRR | 0.6456 +- 0.0185 | 0.6437 +- 0.0199 | -0.0019 | p=0.89 |
| pure hit@1 | 0.1635 +- 0.0328 | 0.1751 +- 0.0400 | +0.0116 | p=0.67 |
| macro hit@1 | 0.0879 +- 0.0112 | 0.0955 +- 0.0170 | +0.0076 | |
| val/loss | 5.4620 +- 0.0352 | 5.4999 +- 0.0780 | +0.0379 | |

**The assembly head does not help retrieval.** Nothing clears significance, and
the per-seed deltas swing both ways (corrected hit@1: +0.008, -0.016, -0.032,
+0.026, +0.008). The single-run +0.0152 reported earlier was noise, exactly as
the earlier num_workers spread suggested it might be.

It does not *hurt* either, and it recovers joint chemistry at 93-99% of headroom
(section 5.6), so it remains defensible as the core-decoration objective -- the
thing that makes a retrieved R-group actually attachable. It is simply not a
retrieval improvement, and the earlier claim that "all three contrastive losses
improved" does not replicate: val/loss is marginally *worse* with the head.

### The scoring correction is the result that matters

Baseline, averaged over 5 seeds:

| K | pure similarity | lift | popularity-corrected | lift | prior |
|---|---|---|---|---|---|
| 1 | 0.1635 | **0.61x** | **0.5872** | **2.20x** | 0.2668 |
| 10 | 0.3457 | 0.51x | 0.7441 | 1.10x | 0.6737 |
| 100 | 0.5793 | 0.71x | 0.8063 | 0.98x | 0.8193 |

Ranked by pure similarity the model sits **below the frequency prior** (0.51-0.71x),
landing inside MolDAM's measured 0.64-0.80x band on a different corpus. Adding
`log p` back at scoring takes hit@1 to **2.20x the prior**. This independently
reproduces MolDAM devlog Phase 9 -> Phase 10: the negative result was a scoring
artifact, not a model failure.

### Removing the condvec fixed the list collapse

| | with condvec | without |
|---|---|---|
| distinct R-groups at rank 1 | 680 | **5,851** |
| library coverage in any top-10 | 11.4% | **100%** |

An ~8.6x increase in retrieved-list diversity. With the leaked condition the
model returned a small pool of common R-groups; without it the lists are genuinely
query-specific.

### What is still not learned

Pure hit@1 by target-frequency bucket: top-10 **0.202**, 10-100 0.126, 100-1k
0.070, 1k-10k **0.036**. Corrected hit@100 sits at 0.98x -- *at* the prior. The
head of the distribution is learned; the tail is not.

---

