# MolPallete

**A MolPLA variant for flavor compound optimization.**

MolPallete pretrains a graph neural network to understand molecular **cores** and
**R-groups** in flavor chemistry, so that it can suggest R-group replacements for a
given core template — lead optimization, applied to taste and aroma rather than to
drugs.

It combines three lineages:

- **[MolPLA](https://doi.org/10.1093/bioinformatics/btae256)** contributes the
  pretraining formulation: masked graph contrastive learning over a molecule and its
  decomposition, with a per-linker R-group retrieval head.
- **MolDAM** contributes the engineering: Hydra config groups, the
  `LightningModule(LossModule(Model))` composition, per-molecule `.pt` corpora with
  in-corpus provenance, and the sample-one-decomposition-per-`__getitem__` idiom.
- **FlavorDB and COCONUT** contribute the chemistry — 25,595 flavor compounds and
  489,395 natural products, in place of MolPLA's drug-like GEOM.

The design contract, every measured number, and the reasoning behind each choice
live in **[`molpallete/docs/molpallete_design.md`](molpallete/docs/molpallete_design.md)**.

---

## Repo layout

```
molpallete_preprocess/       corpus builder      (see its own README)
  preprocess_flavor.py         corpus driver
  enumerate_rgroups.py         R-group library vocabulary builder
  molpallete_prep/             the package
  scripts/build_all_corpora.sh
  scripts/build_all_vocabs.sh
molpallete/                  training repo
  docs/molpallete_design.md    design contract + open items
  docs/molpallete_corpus_eda.pdf  corpus statistics (regenerate: generate_eda_report.py)
  docs/archive/                superseded results, with what changed under them
  src/configs/                 Hydra config groups
  src/data_modules/            four-view dataset + collate
  src/nnet_modules/            encoder + projection heads + composite model
  src/loss_modules/            dual InfoNCE + the three-objective module
  src/lightning_modules/       training wrapper
  src/callbacks/               FAISS retrieval, full-library RGR, prediction tables, health
  src/build_library.py         export a trained FAISS R-group library for inference
```

---

## What MolPallete changes, and why

### 1. It restores MolPLA's per-linker retrieval head

MolDAM replaced MolPLA's per-linker query→R-group contrastive objective with a
sum-pooled R-group *bag* contrasted against a single core. That collapses
cardinality — four R-groups pointing the same way become indistinguishable from
one. MolPallete restores the per-linker form, which is what makes "suggest a
replacement **at this position**" a well-posed query.

It also restores the `sg_P` / `sg_R` / `sg_Q` stop-gradient toggles, which MolDAM
dropped. All three MolPallete objectives are contrastive with no negative-free
branch, so these are the only collapse-mitigation lever available.

### 2. It fixes a decomposition that is degenerate on flavor chemistry

Measured on 400 FlavorDB molecules, `naveja_recap` — *MolPLA's own method* — yields
**1.04 R-groups per decomposition**, with only 3.7% of decompositions having two or
more. MolPLA's `islinked` subset space is `2^k − 1`, so at `k = 1` there is exactly
one instance per core and the core-decoration objective has nothing to decorate.
This reproduces the same structural failure MolDAM_prep measured on ZINC (99.42%
single-R-group).

MolPallete's fix is to re-frame multi-cut decompositions as anchored stars. A
partition induces a **fragment tree**; any connected subset of it taken as the core
leaves one R-group per boundary edge. Enumerating connected subsets above a size
ratio reproduces MolPLA's "single molecule, multiple putative cores" property:

| method | cores/mol | R-groups/core | k ≥ 2 | core heavy atoms |
|---|---|---|---|---|
| `naveja_recap` (ratio 1/2) | 12.81 | 1.06 | 5.5% | — |
| **`macfrag` re-framed (ratio 1/2)** | 64.45 | **3.34** | **89.7%** | **22.5** |
| MolPLA, as reported on GEOM | 4.04 | — | — | ~20.8 |

`--max-cores 10` prunes the candidate surplus, as MolPLA did.

### 3. It conditions retrieval on flavor chemistry, not bulk thermodynamics

MolPLA conditions R-group retrieval on a functional-group vector computed with
`thermo`, whose checks are bulk-thermodynamic classes. MolPallete uses 85 RDKit
`fr_*` counters plus 12 SMARTS patterns central to flavor and aroma —
pyrazines, pyrroles, di/trisulfides, acetals, isoprene units — for a 97-bit vector.
A **`pocket` mode** is declared with a fixed interface but ships unfed; it returns
zeros, which is deliberately the paper's `Cond. None` ablation, so an unfed pocket
run shows collapsed retrieval rather than plausible-looking noise.

---

## Quick start

```bash
# 1. Build a corpus (FlavorDB, ~14 s on 88 workers)
cd molpallete_preprocess
python preprocess_flavor.py --source flavordb --method macfrag \
  --output-path /home/mogan/preprocessed/molpallete/flavordb_full/macfrag --workers 88

# ... or build every corpus
CORPORA=/home/mogan/preprocessed/molpallete ./scripts/build_all_corpora.sh

# 2. Build the R-group library vocabulary (the RGR retrieval target space)
python enumerate_rgroups.py --corpus /home/mogan/preprocessed/molpallete/flavordb_full/macfrag --workers 88

# 3. Sanity-check the training loop (~40 s, CPU)
cd ../molpallete/src
python run.py --config-name config_debug \
  trainer_kwargs.fast_dev_run=1 trainer_kwargs.accelerator=cpu \
  data_module_kwargs.dataset_path=/home/mogan/preprocessed/molpallete \
  data_module_kwargs.num_workers=0 data_module_kwargs.persistent_workers=false

# 4. Pretrain
python run.py --config-name config \
  data_module_kwargs.dataset_path=/home/mogan/preprocessed/molpallete \
  data_module_kwargs.dataset_version=flavordb_full \
  data_module_kwargs.decomposition_method=macfrag \
  trainer_kwargs.accelerator=gpu trainer_kwargs.devices=1 \
  trainer_kwargs.precision=bf16-mixed \
  experiment_name=flavor_macfrag_v1

# 5. Export the trained library for lead-optimization queries
python build_library.py \
  --checkpoint /home/mogan/checkpoints/flavor_macfrag_v1_best.pt \
  --corpus /home/mogan/preprocessed/molpallete/flavordb_full/macfrag \
  --config /home/mogan/preprocessed/molpallete/logs/pretrain_v1/.hydra/config.yaml \
  --output /home/mogan/libraries/flavor_macfrag_v1
```

## The R-group library

MolPLA's retrieval task scores against **every recommendable R-group in the
corpus**, not against in-batch negatives — that library is the lead-optimization
task. MolPallete builds it in two halves: a static per-corpus **vocabulary**
(`enumerate_rgroups.py`), and a **vector library** re-embedded every validation
epoch because the projector is still training.

The FlavorDB macfrag vocabulary has 5,663 distinct R-groups over 399,141
occurrences — but an **effective size of 31**, with one R-group accounting for
19% of occurrences. So every `hit@K` is logged beside `prior_hit@K` (the constant
"return the K most frequent" predictor) and their ratio `lift@K`. On an untrained
model, `hit@1000 = 0.098` against a prior of `0.947`: a number that looks
non-trivial in isolation and is in fact far below the baseline. `lift <= 1` means
the model has learned nothing the prior does not already give you.

25 epochs of pretraining on `flavordb_full/macfrag`, scored over all 5,663 R-groups:

| metric | untrained | epoch 7 | epoch 25 | prior | lift |
|---|---|---|---|---|---|
| MRR | 0.0016 | 0.497 | **0.723** | — | — |
| Hit@1 | 0.000 | 0.374 | **0.645** | 0.140 | **4.6×** |
| Hit@10 | 0.005 | 0.675 | **0.847** | 0.650 | **1.30×** |
| Hit@100 | 0.008 | 0.798 | **0.952** | 0.823 | **1.16×** |

On the **combined corpus** `coconut-flavordb_full` (369,881 molecules, a 60,773-R-group library at
effective size 34 — the same scale as MolPLA's 61,279 on GEOM), 30 epochs:

| metric | epoch 1 | epoch 30 | prior | lift |
|---|---|---|---|---|
| MRR | 0.625 | **0.811** | — | — |
| Hit@1 | 0.535 | **0.741** | 0.265 | **2.80×** |
| Hit@10 | 0.768 | **0.933** | 0.671 | **1.39×** |
| Hit@100 | 0.873 | **0.987** | 0.809 | **1.22×** |

It clears the prior at every cut-off from epoch 1, unlike the FlavorDB-only run.

On FlavorDB alone, at epoch 7 the model beat the prior at Hit@1 but was *at or below* it at Hit@10
and Hit@100 — top-of-list ranking learned, tail not yet. By epoch 25 it clears
the prior everywhere. These are not comparable to MolPLA's published numbers
(MRR 0.2616 on GEOM): that library has 61,279 distinct R-groups against this
one's 5,663, so a higher MRR here reflects an easier library, not a better model.

## Corpora built

| method | records | dec/mol | R-groups/dec | library | effective |
|---|---|---|---|---|---|
| **`macfrag`** *(default)* | 369,881 | 7.57 | **3.12** | 48,671 | 32 |
| `naveja_recap` | 411,456 | 8.20 | 1.08 | 113,362 | 283 |
| `bemis_murcko` | 299,868 | 1.00 | 3.64 | 22,814 | 30 |
| `synton` | 168,981 | 3.15 | 1.32 | 34,557 | 744 |

All four are built from `coconut-flavordb_full`, the single combined corpus
(351,382 COCONUT + 18,499 FlavorDB after deduplication). Each has its R-group
library vocabulary built and hash-version matched.

Note the anti-correlation between the last three columns: `macfrag` gives the
richest decoration (3.12 R-groups per decomposition) but the most degenerate
retrieval library (effective size 32), while `synton` is the reverse. The same
property that makes decoration learnable makes retrieval easier to fake, so
`macfrag` retrieval numbers must be read against their frequency prior.


| **`coconut-flavordb_full/macfrag`** *(primary)* | **369,881** | **7.57** | **3.12** | 4.3 GB |
| `coconut_full/macfrag` | 354,515 | 7.56 | 3.13 | 4.1 GB |

`synton` retained 48.3% of molecules, matching its measured 50.2% no-partition
rate on flavor chemistry (it is also ~30× slower than every other method).
COCONUT's natural products are larger and more decorated than flavor volatiles,
so they yield *more* multi-R-group structure than FlavorDB — which is exactly the
property MolPLA's objectives need.

## Environment notes

Built and measured on 96 cores / 354 GB RAM / 2× RTX PRO 6000 Blackwell, conda env
`maven`, Python 3.12.13. Two host-specific gotchas:

- **`faiss-gpu-cu12` 1.14.1 has no Blackwell (sm_120) kernels.** `FAISSRetrieval`
  therefore defaults to `index_type="flat_cpu"`.
- **`thermo` is not installed**, which is why the condition vector is RDKit-derived.

## Citation

```bibtex
@article{gim2024molpla,
  title   = {MolPLA: A Molecular Pretraining Framework for Learning Cores, R-Groups and Linkers},
  journal = {Bioinformatics},
  volume  = {40}, number = {Supplement 1}, pages = {i369--i380}, year = {2024},
  doi     = {10.1093/bioinformatics/btae256}
}
```
