# MolPallete — Design Notes (working draft)

> Status: **v1 built and validated end-to-end.** Corpora built, forward/backward
> verified, full Hydra training loop runs. Written from the MolPLA paper + released
> code, MolDAM, and MolDAM_prep, with every claim below measured on this machine.

MolPallete = **MolPLA's pretraining objectives** + **MolDAM's engineering** + **flavor chemistry**.

---

## 1. What MolPallete is

1. A variant of MolPLA with an **auxiliary pretraining objective for lead optimization
   (core decoration)**.
2. Pretrained on **flavor compounds** (FlavorDB) and **natural products** (COCONUT),
   not on GEOM/ZINC drug-like space.
3. The **condition vector** is either **protein-pocket context** or **neutral
   (ligand-only)** — replacing MolPLA's fixed 88-dim R-group functional-group vector.
4. Shares MolDAM's **anchored paradigm** (core + K R-groups, masked clone-joints,
   multi-decomposition enumeration per molecule).

Inheritance rule from the brief: `molpallete/` inherits **the primary traits of MolPLA,
not MolDAM** — MolDAM contributes code structure, implementation style, and the
efficient decomposition/corpus machinery.

---

## 2. MolPLA — the traits to inherit (verified against source)

Source: `/home/mogan/github/MolPLA` (`src/models/MolPLA.py`, `src/dataloaders/base.py`,
`src/utils.py`, `settings.yaml`).

### 2.1 The three views — paper notation vs code notation

The paper and the reference implementation use **different letters for the same objects**.
MolPallete standardises on the code's letters and records the mapping once:

| Code (`MolPLA.py`) | Paper (btae256) | Content |
|---|---|---|
| `G` | `𝒢_M` | the intact molecule graph, all linker flags cleared |
| `P` | `𝒢_{Q_{i,j,k}}` — the *query template* | core `𝒢_C` ∪ the R-groups **not** decoupled; one masked linker atom per *decoupled* R-group |
| `R` | `𝔾_{R_{i,j,k}}` | the decoupled R-groups, one graph each, each with one masked linker atom |
| `Q` | `𝒢_{D_{i,j,k}}` | `P ∪ R` — node-set union, pooled as one graph |

⚠️ The paper's `Q` is the code's `P`. Do not mix them. MolPallete uses **G / P / R / Q** as above.

`k` indexes a **non-empty subset** of the core's R-groups; the code stores it as the
`islinked` bitstring (`'1'` = stays attached → part of `P`, `'0'` = decoupled → a member of
`R`). The paper has no name for the bitmask, only the subset index `k`.

All three views are encoded by **one shared encoder in a single forward pass**: `G`, `P` and
every `R` are collated into one `W_data` and split afterwards by boolean masks
`G_markers` / `P_markers` / `R_markers` (`convert_molpla_data`, base.py:302). This is the
trait that makes MolPLA cheap — one GNN call, not three.

**Masking rule** (paper §2.1): only the linker joints of *decoupled* R-groups are masked.
A linker joint whose R-group is still attached stays intact. Masking covers **both node and
edge attributes**, which is what frees retrieval from matching the original linker chemistry.

**One linker atom is shared** between core and R-group — MolPLA does *not* cut between two
atoms and add dummies. MolDAM's `detach_rgroups_multi` implements exactly this: the core-side
atom is punched (masked in place) and a masked *clone* is appended to the R-group.

### 2.2 The three contrastive losses

All use the same `ContrastiveLoss(score_func='dualentropy')` (`src/utils.py:12`):
symmetric in-batch InfoNCE on L2-normalised views —
`0.5·CE(XYᵀ/τ, I) + 0.5·CE(YXᵀ/τ, I)`, cosine similarity. The paper calls this
"Dual InfoNCE" (Eq. 13) and never uses the phrase "dual entropy" — that is a code-only name.

| # | Name | Pair | τ (paper) | τ (code) | coef (paper) | coef (code) |
|---|---|---|---|---|---|---|
| 1 | `graph_contrastive` | pooled `G` ↔ pooled `Q` | 0.01 | 0.1 | 1.0 | 1.0 |
| 2 | `linker_contrastive` | node emb at each linker in `G` ↔ `P_lj + R_lj` | 0.05 | 0.05 | 1.0 | 0.1 |
| 3 | `rgroup_contrastive` | query = `[P linker node emb ‖ condvec]` ↔ pooled `R` | 0.01 | 0.01 | 1.0 | 1.0 |

Batch sizes differ per loss: `B1` = #instances, `B2` = #linker nodes, `B3` = #(query, R-group)
pairs, with `B1 ≥ B2` and `B1 ≥ B3`.

Loss #3 is the **R-group retrieval / lead-optimization** head — it is *per-linker*, not a
sum-pooled bag. This is precisely the MolPLA trait MolDAM dropped (MolDAM anchored
§2, §5.1) and the one MolPallete must restore.

### 2.3 Stop-gradient toggles

`sg_P`, `sg_R`, `sg_Q` (`settings.yaml`, default `False/False/True`) detach the
`P`/`R`/`Q` branches — SimSiam-style collapse mitigation. MolDAM has none. **Keep these.**

### 2.4 Condition vector

MolPLA: a binary functional-group presence vector of the *target R-group*, concatenated
onto the core-linker node embedding before the query projector (`prop_conditioned: rgroups`).

⚠️ **Dimension disagrees between paper and code**: the paper states `c_R ∈ [0,1]^87`
(Eq. 11, verified at glyph level); the released code allocates **88** (`np.zeros((n, 88))`,
base.py:406) and fills it from `thermo.functional_groups` (`get_molecular_functional_groups`,
base.py:58). `thermo` is **not installed** in the `maven` env.

Ablations in the paper show this vector is load-bearing: `Cond. None` (all-zero) collapses
retrieval MRR from 0.2616 to 0.0056, and `Cond. All` (all-one) to <0.0001 — a degenerate
condition is *worse* than no condition.

MolPallete generalises this slot to `{neutral, pocket}` — see §5.

### 2.5 Data-instance schema (per molecule, pre-transform)

```
data_instance_id            "<mol_id>-<islinked bitstring>"
complete_mol_nx             nx.Graph of G, per-node linker_info
incomplete_mol_nx           nx.Graph of P
disjoint_rgroups_nx         list[nx.Graph], one per R-group
islinked_rgroups            bitstring; '1' = stays on core, '0' = detached target
smiles_original / _masked_core / _masked_rgroups[] / _partially_assembled
```

One molecule with k R-groups yields up to `2^k − 1` instances (one per non-trivial
`islinked` mask). This is where the **core-decoration** objective naturally lives.

---

## 3. MolDAM — what to borrow (structure, not objectives)

Source: `/home/mogan/github/MolDAM`, `/home/mogan/github/MolDAM_prep`.

- Hydra config groups (`configs/{api,data_module,nnet_module,loss_module,lightning,trainer,wandb}/`)
- `LightningModule(LossModule(Model))` composition
- `VanillaGNN` encoder: per-attribute `nn.Embedding` → fusion MLP → `GINEConv` blocks
- Per-mol `.pt` corpus with `shard2`/`shard3` layout + `__meta__.json` / `__manifest__.json`
- MolPLA-style **sampling semantics**: `__len__` = #molecules; `__getitem__` draws one
  random decomposition
- Callbacks: `FAISSRetrieval`, `MolecularReassembly`, `PredictionTable`, `SaveBestModelCheckpoint`
- `preprocess_*.py` worker-pool driver, seeded per-record Bernoulli sampling, atomic
  metadata writes, refuse-to-overwrite

### 3.1 Hard-won findings from MolDAM_prep that constrain MolPallete

1. **`naveja_recap` with a `ratio` threshold cannot produce multi-R-group
   decompositions** — 99.42% of instances have exactly one R-group at `ratio=2/3`, and
   even `ratio=1/2` only reaches 0.83%. Cause is structural: `find_putative_cores` takes
   RECAP *children* as cores, and a child is one connected fragment. A 5.9 h rebuild was
   spent discovering this.
   **Consequence for MolPallete:** MolPLA's whole formulation needs `k ≥ 2` R-groups for
   the `islinked` bitmask and core-decoration objective to be non-degenerate. So the
   corpus must come from a **multi-cut** decomposer (`macfrag`, `synton`) re-framed into
   core+R-groups, or from `naveja` at a much lower ratio. **This is the central
   preprocessing design decision.** `TODO`: confirm against the decomposer specs.
2. **`synton` yields no partition for 21% of molecules** and produces the largest
   fragments (7.27 heavy atoms mean, 4.04 frags/mol).
3. **`macfrag`** gives 8.01 frags/mol at 4.67 heavy atoms — the best multi-cut candidate.
4. Vocabulary skew differs wildly by paradigm; any retrieval metric must be reported
   against the **frequency prior**, not against random (MolDAM devlog Phase 11).
5. Use `--layout shard3` for anything above ~10% scale.

---

## 3.2 Corpus naming

`{sources}_{scale}`, sources joined by `-` in the order they are read. Mirrors
MolDAM's `{scale}` slot convention (`zinc_1pct`), so a subsampled build would be
`coconut-flavordb_1pct`.

| corpus | sources | records | note |
|---|---|---|---|
| `coconut-flavordb_full` | both | 369,881 (macfrag) | the only corpus |

Per-source corpora (`coconut_full`, `flavordb_full`) and the pre-hash-fix
`coconut-flavordb_full_hashv1` were **deleted** on 2026-08-19: the source
ablation is not a planned experiment, and the hash comparison in §5 is already
recorded. All three are derived data and rebuildable from
`scripts/build_combined.sh` against the untouched source datasets, at roughly
5 min per corpus plus 4 min per vocabulary.


Each holds one sub-directory per decomposition method (`macfrag`,
`naveja_recap`, `bemis_murcko`, `synton`). `_full` means no subsampling; the
heavy-atom filter `[5, 50]` and COCONUT variant deduplication still apply and are
recorded in each corpus's provenance block.

The two per-source corpora predate always-on deduplication, so they are not
directly comparable to the combined ones on record counts.

---

## 4. Repo layout

```
MolPallete/
  molpallete_preprocess/         corpus builder   (mirrors MolDAM_prep)
    preprocess_flavor.py           the driver
    molpallete_prep/               the package
    scripts/build_all_corpora.sh
    docs/corpus_format.md
  molpallete/                    training repo    (mirrors MolDAM)
    docs/molpallete_design.md      this file
    src/{configs,data_modules,nnet_modules,loss_modules,lightning_modules,callbacks,scripts}
```

The brief names the preprocessing directory `molpallete_prep`; the pre-created
directory was `molpallete_preprocess`, so that is the repo name and
`molpallete_prep` is the Python package inside it.

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
Implemented in `molpallete_prep/anchored_from_partition.py`.

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

MolPallete builds it in two halves, because they have different lifetimes:

| half | built by | depends on | rebuilt |
|---|---|---|---|
| **vocabulary** — distinct R-groups, canonical masked graphs, counts, condvecs | `molpallete_preprocess/enumerate_rgroups.py` | the corpus only | once per corpus |
| **vector library** — those graphs embedded + FAISS index | `callbacks/RGroupLibraryRetrieval.py` (training) / `molpallete/src/build_library.py` (inference) | the projector's current weights | **every validation epoch** |

The second half must be rebuilt continuously: a library embedded at epoch 3 is
meaningless for a query embedded at epoch 7. MolPLA rebuilds it at every
validation; so does the callback.

**Keying.** MolPLA keys the vocabulary on the R-group's masked SMILES. MolPallete
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

## 6. Design decisions and their reasons

| Decision | Reason |
|---|---|
| `macfrag` is the default method | the only one giving MolPLA-shaped multi-R-group cores on flavor chemistry (§5.1–5.2) |
| stereochemistry preserved (`wash(remove_stereo=False)`) | (Z)- and (E)-3-hexenol are different odorants; MolDAM_prep discarded stereo unconditionally |
| condition vector is 97-dim, RDKit-derived | `thermo` (MolPLA's source) is not installed and its 88 checks are bulk-thermodynamic classes, not flavor chemistry. 85 RDKit `fr_*` counters + 12 flavor SMARTS (pyrazines, di/trisulfides, acetals, isoprene units) |
| pocket condvec returns zeros | reproduces the paper's `Cond. None` ablation exactly, so an unfed pocket run shows collapsed retrieval instead of plausible noise |
| only bookkeeping stored, not detached graphs | materialising all `2^k − 1` subsets would multiply the corpus ~9x; the dataset detaches at `__getitem__` |
| joints paired by `linker_id`, not by position | MolPLA's positional bookkeeping is implicit and fragile; `build_instance` returns explicit `joint_linker_ids` / `joint_G_atoms` |
| projectors shared across both sides of losses 1 and 2 | an asymmetric projection lets the branches drift into separate subspaces and satisfy the objective without the encoder learning anything shared |
| multi-positive masking + logQ correction on the R-group loss | measured `dup_rate = 0.69`; and MolDAM's headline retrieval number, re-scored against the frequency prior rather than random, landed *at* the prior |
| `hash3` shard layout | FlavorDB ids are variable-width (`FDB4` … `FDB25595`), so suffix-slicing buckets them on letters |
| metadata written atomically | MolDAM_prep's clearest unfixed defect; `__meta__.json` reaches tens of MB and a torn write destroys the index |

---

## 7. Open items

- **Pocket condition vector is unfed.** The interface is fixed
  (`PocketCondVec.encode(mol, context)`); no pocket source is wired. Candidates:
  taste/olfactory receptors (not on disk, and flavor molecules have no measured
  receptor pairings) or CrossDocked2020 (on disk, but drug targets).
- **logQ estimator consistency.** The loss estimates `log q` from a running
  in-batch counter; `FAISSRetrieval` estimates it from val-split hash frequency.
  If those disagree the correction is inconsistent between train and eval.
- **`vocab.py` / `lmdb_store.py`** are carried over from MolDAM_prep and are now
  superseded by `rgroup_library.py` for vocabulary purposes; they remain unwired.
- **No pretraining run at scale yet.** The single-batch overfit check drives
  val/loss 15.28 -> 7.79 and val R@10 0.378 -> 0.647 over 60 epochs, which
  establishes that the objectives carry gradient signal but says nothing about
  what the representation learns at corpus scale.
- **`faiss-gpu-cu12` 1.14.1 has no Blackwell (sm_120) kernels** on this host, so
  `FAISSRetrieval` defaults to `index_type="flat_cpu"`. Fine at the current
  gallery size; revisit if the gallery reaches millions of rows.
- **Corpora built before 2026-08-18 06:16 lack `rdkit_version`** in their
  provenance block. The five completed at that point were backfilled atomically
  from the same interpreter that built them; the field is written natively from
  then on.

---

## 8. Environment (measured)

96 cores · 354 GB RAM · 2× RTX PRO 6000 Blackwell (97 GB each) · 1.0 TB free on `/`.
Conda env `maven`, Python 3.12.13. torch 2.11.0+cu130, torch_geometric 2.7.0,
lightning 2.6.1, rdkit 2026.3.2, hydra-core 1.3.2, lmdb, zstandard, wandb,
faiss-gpu-cu12 1.14.1, python-igraph 1.0.0. **`thermo` is not installed** — see §6.
