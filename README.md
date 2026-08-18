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
live in **[`docs/molpallete_design.md`](docs/molpallete_design.md)**.

---

## Repo layout

```
docs/molpallete_design.md    design contract + measured results + open items
molpallete_preprocess/       corpus builder      (see its own README)
  preprocess_flavor.py         the driver
  molpallete_prep/             the package
  scripts/build_all_corpora.sh
molpallete/                  training repo
  src/configs/                 Hydra config groups
  src/data_modules/            four-view dataset + collate
  src/nnet_modules/            encoder + projection heads + composite model
  src/loss_modules/            dual InfoNCE + the three-objective module
  src/lightning_modules/       training wrapper
  src/callbacks/               FAISS retrieval, prediction tables, representation health
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
  --output-path /home/mogan/corpora/molpallete/flavor_v1/macfrag --workers 88

# ... or build every corpus
CORPORA=/home/mogan/corpora/molpallete ./scripts/build_all_corpora.sh

# 2. Sanity-check the training loop (~40 s, CPU)
cd ../molpallete/src
python run.py --config-name config_debug \
  trainer_kwargs.fast_dev_run=1 trainer_kwargs.accelerator=cpu \
  data_module_kwargs.dataset_path=/home/mogan/corpora/molpallete \
  data_module_kwargs.num_workers=0 data_module_kwargs.persistent_workers=false

# 3. Pretrain
python run.py --config-name config \
  data_module_kwargs.dataset_path=/home/mogan/corpora/molpallete \
  data_module_kwargs.dataset_version=flavor_v1 \
  data_module_kwargs.decomposition_method=macfrag \
  trainer_kwargs.accelerator=gpu trainer_kwargs.devices=1 \
  trainer_kwargs.precision=bf16-mixed \
  experiment_name=flavor_macfrag_v1
```

## Corpora built

| corpus | records | dec/mol | R-groups/dec | size |
|---|---|---|---|---|
| `flavor_v1/macfrag` *(default)* | 20,428 | 7.27 | 2.69 | 242 MB |
| `flavor_v1/naveja_recap` | 23,216 | 8.76 | 1.07 | 231 MB |
| `flavor_v1/bemis_murcko` | 19,121 | 1.00 | 5.17 | 166 MB |
| `flavor_v1/synton` | 12,084 | 2.52 | 1.12 | 111 MB |

`synton` retained 48.3% of molecules, matching its measured 50.2% no-partition
rate on flavor chemistry (it is also ~30× slower than every other method).

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
