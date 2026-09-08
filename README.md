# MolPLAtte

**A MolPLA variant for flavor compound optimization.**

MolPLAtte pretrains a graph neural network to understand molecular **cores** and
**R-groups** in flavor chemistry, so that it can suggest R-group replacements for a
given core template — lead optimization, applied to taste and aroma rather than to
drugs. A second stage conditions that retrieval on a receptor pocket.

It combines three lineages:

- **[MolPLA](https://doi.org/10.1093/bioinformatics/btae256)** contributes the
  pretraining formulation: masked graph contrastive learning over a molecule and its
  decomposition, with a per-linker R-group retrieval head.
- **MolDAM** contributes the engineering: Hydra config groups, the
  `LightningModule(LossModule(Model))` composition, per-molecule `.pt` corpora with
  in-corpus provenance, and the sample-one-decomposition-per-`__getitem__` idiom.
- **FlavorDB and COCONUT** contribute the chemistry — 25,595 flavor compounds and
  489,395 natural products, in place of MolPLA's drug-like GEOM.

The design contract and the reasoning behind each choice live in
**[`molplatte/docs/molplatte_design.md`](molplatte/docs/molplatte_design.md)**.
Measured results live beside them in `molplatte/docs/`, with superseded ones in
`docs/archive/` and a note on what changed.

---

## Repo layout

```
molplatte_preprocess/            corpus builder (see its own README)
  preprocess_flavor.py             corpus driver -- all four sources
  enumerate_rgroups.py             R-group library vocabulary builder
  src/molplatte_prep/              the package (decomposition, condvec, readers,
                                   pocket extraction, graph hashing)
  tests/                           54 regression guards for SILENT failures
  scripts/                         build, rebuild, annotate, report
  docs/                            corpus format, pocket guide, EDA reports

molplatte/                       training repo
  src/configs/                     Hydra config groups
  src/data_modules/                four-view dataset + collate + CV folds
  src/nnet_modules/                encoder, projection heads, composite model
    components/pocket_conditioning.py  pocket half of the condition vector
  src/loss_modules/                dual InfoNCE + the three-objective module
  src/lightning_modules/           training wrapper
  src/callbacks/                   FAISS retrieval, full-library RGR,
                                   prediction tables, representation health
  src/lead_optimization.py         inference: any molecule, any structure
  src/build_library.py             export a trained FAISS R-group library
  src/scripts/                     experiment drivers and result summarisers
  notebooks/lead_optimization.ipynb  interactive lead optimization
  docs/                            design contract + measured results
```

Corpora and run outputs live outside the repo, under
`~/preprocessed/molplatte/` and `~/checkpoints/molplatte/`.

---

## What MolPLAtte changes, and why

### 1. It restores MolPLA's per-linker retrieval head

Retrieval is scored against **every recommendable R-group in the corpus**, not
against in-batch negatives. That library *is* the lead-optimization task.

### 2. It fixes a decomposition that is degenerate on flavor chemistry

The default is `naveja_recap` at `ratio=1/3`, ring-aware, `max_cores=4`,
`min_rgroup_atoms=2`. Earlier work defaulted to `macfrag`, which gives richer
decoration but a far more degenerate retrieval library — the same property that
makes decoration learnable makes retrieval easy to fake.

### 3. The condition vector is exogenous, and that is the whole point

MolPLA conditions the query on a functional-group vector **of the target
R-group** (paper Eq. 11), so the query contains a description of the answer.
Measured here on a model trained that way: zeroing the condvec at inference took
hit@1 from 0.6111 to **exactly 0.0000**, and MolPLA's own `Cond. None` ablation
collapses its MRR 0.2616 → 0.0056. A 47× drop from removing an "auxiliary" hint
means it was never auxiliary.

MolPLAtte's condition vector is therefore **24 flavour bits sourced from
measurement, literature and physics — never from the molecular graph**:

| source | rule |
|---|---|
| measured | FlavorDB and curated sensory databases, joined by InChIKey |
| mined | LLM annotations at the `documented` tier only, with a named source |
| physics | `odorless` for compounds too heavy to volatilise (MW > 350) |
| — | everything else gets `unknown`; an all-zero vector is never emitted |

`structural` and `close_analog` annotation tiers are deliberately excluded: both
are inferred from the graph and would reintroduce the leak.

---

## Quick start

```bash
# 1. Build a corpus (FlavorDB only, ~20 s on 96 workers)
cd molplatte_preprocess
export PYTHONPATH=$PWD/src
A=~/preprocessed/molplatte/annotations

python preprocess_flavor.py \
  --source flavordb --method naveja_recap \
  --output-path ~/preprocessed/molplatte/flavordb-only.new \
  --core-ratio 0.3333333333333333 --max-cores 4 --max-rgroups 8 \
  --min-rgroup-atoms 2 --keep-stereo --no-neutralise \
  --min-heavy-atoms 5 --max-heavy-atoms 50 \
  --condvec-mode flavor \
  --flavor-measured $A/flavor_measured.jsonl \
  --flavor-mined    $A/flavor_documented.jsonl \
  --layout hash3 --workers 96

# --output-path writes records WITHOUT a <method>/ level; the rest of the
# tooling expects <corpus>/naveja_recap/, so restructure:
mkdir -p ~/preprocessed/molplatte/flavordb-only/naveja_recap
mv ~/preprocessed/molplatte/flavordb-only.new/* \
   ~/preprocessed/molplatte/flavordb-only/naveja_recap/

# ... or build all four named corpora at once
bash scripts/rebuild_named_corpora.sh

# 2. Build the R-group library vocabulary (the retrieval target space)
python enumerate_rgroups.py \
  --corpus ~/preprocessed/molplatte/flavordb-only/naveja_recap --workers 96

# 3. Run the regression guards (~12 s)
python -m pytest tests -q

# 4. Pretrain (STEP 1: ligand-only, flavour condition)
cd ../molplatte/src
CUDA_VISIBLE_DEVICES=0 python run.py \
  experiment_name=s1-pretrain \
  data_module_kwargs.dataset_version=coconut-flavordb-full \
  data_module_kwargs.condvec_dim=24 \
  nnet_module_kwargs.condvec_dim=24 \
  rgroup_library.vocab_path=~/preprocessed/molplatte/union_vocab/base-full__crossdocked__tastepocket/rgroup_vocab.pkl.gz \
  ++trainer_kwargs.max_epochs=30 \
  wandb.project=molplatte

# 5. Pocket-conditioned finetuning (STEP 2: 5-fold CV, then a full-data model)
bash scripts/run_step2_pocket_cv.sh
python scripts/summarise_step2_cv.py
```

For lead optimization against a trained checkpoint, open
[`molplatte/notebooks/lead_optimization.ipynb`](molplatte/notebooks/lead_optimization.ipynb).

---

## The R-group library

The library is built in two halves: a static per-corpus **vocabulary**
(`enumerate_rgroups.py`), and a **vector library** re-embedded every validation
epoch, because the projector is still training and an index built by an earlier
checkpoint scores against a different space.

Every `hit@K` is logged beside `prior_hit@K` — the constant "return the K most
frequent R-groups" predictor — and their ratio `lift@K`. **`lift <= 1` means the
model has learned nothing the prior does not already give you.** Reading hit@K
alone is how a degenerate library flatters a model.

The right size measure is **effective vocabulary** (`exp(H)` of the frequency
distribution): how many-way retrieval actually is, as opposed to the row count.
The primary library has 86,385 rows and an effective size of 889.

### Pretraining, `coconut-flavordb-full`, 30 epochs

Scored against the 91,935-row union library (effective size 946):

| metric | value | prior | lift |
|---|---:|---:|---:|
| MRR | **0.4247** | — | — |
| Hit@1 | **0.3212** | 0.0248 | **13.0×** |
| Hit@10 | **0.6315** | 0.0898 | 7.0× |
| Hit@100 | **0.8434** | 0.3985 | 2.1× |

Not comparable to MolPLA's published MRR 0.2616 on GEOM: different library,
different chemistry, different prior. Compare against the prior in the same row.

### Union library and the base/novel split

A model scored only against its pretraining vocabulary silently drops every
target it has never seen — `rows_of() == -1`, counted as `n_target_missing`, and
those dropped targets are exactly the rare novel tail. So pocket-stage runs score
against a **union** of the pretraining, CrossDocked and tastepocket vocabularies
(91,935 rows, 5,550 novel) and report `base_hit@K` and `novel_hit@K` separately.

---

## Corpora

All built with `naveja_recap`, `HASH_VERSION 4`, `condvec_version 3`.

| corpus | records | dec/mol | R-groups/dec | library | effective |
|---|---:|---:|---:|---:|---:|
| **`coconut-flavordb-full`** *(primary)* | **393,066** | 2.91 | 1.40 | 86,385 | 889 |
| `coconut-only` | 374,454 | 2.90 | 1.41 | 82,312 | 853 |
| `coconut-flavordb-filtered` | 50,933 | 2.68 | 1.14 | 13,178 | 467 |
| `flavordb-only` | 20,812 | 3.17 | 1.20 | 6,182 | 299 |
| `crossdocked` | 10,689 | 3.00 | 1.34 | 9,435 | 1,478 |
| `tastepocket_corpus` | 243 | 3.02 | 1.23 | 263 | 129 |

`coconut-flavordb-filtered` keeps the COCONUT molecules inside a measured
odorant envelope; the rationale, with every number recomputed from source, is in
[`docs/coconut_filtering_rationale.pdf`](molplatte_preprocess/docs/coconut_filtering_rationale.pdf).

CrossDocked's effective vocabulary (1,478) exceeds the primary corpus's (889)
despite 37× fewer records — its R-groups are spread more evenly, so **retrieval
there is harder**. Numbers are not comparable across corpora in either direction.

---

## Pocket conditioning

STEP 2 conditions retrieval on a receptor pocket: frozen ESM-2 650M embeddings,
mean-pooled over residues within 10 Å of the ligand, reduced inside the model by
`PocketConditioning`. The condition vector becomes `[24 flavour | 1280 pocket]`,
stored raw and projected during training so the reduction is learned.

The fine-tuning set is 343 T1 chemosensory complexes from the PDB → 269
(ligand, receptor) records → 243 that decompose. Evaluation is 5-fold CV cutting
whole connected components of the (ligand, receptor) bipartite graph, because
pooled pocket embeddings identify the receptor family with **98.5% 1-NN accuracy**
and a random split would report memorisation as generalisation.

**The result is negative, and measured two ways:**

| comparison | mean ΔH@1 |
|---|---:|
| pocket on vs pocket off | **+0.0008** |
| finetuned vs not finetuned at all | **−0.0106** |

The pocket is *not* being ignored — it contributes 48% as much as the flavour
half to the query vector and takes 241 distinct values across 906 rows. It is
present, substantial, and carries nothing retrieval can use. Full write-up,
including what the result does and does not license, in
[`molplatte/docs/step2_pocket_results_2026-09-05.md`](molplatte/docs/step2_pocket_results_2026-09-05.md).

The likeliest cause is scale: 269 records cannot separate a pocket→chemistry
mapping from receptor identity, which is useless on a held-out receptor by
construction.

---

## Tests

`molplatte_preprocess/tests/` holds 54 guards, every one written for a bug that
**ran to completion and wrote plausible output**. None would have been caught by
"does it crash?":

- the flavour condvec collapsed to a single MW>350 bit across all four corpora
- `uint8` storage wrapped ESM embeddings to `{0, 250, 255}` — right shape, right
  dtype, no information
- `resname TRP` matched 14 backbone residues per structure as "ligands"
- CrossDocked's official test set leaked into train through ligand dedup
- `read_coconut` raised `NameError` the moment its stream ended

CI runs them on every push, plus a **mutation check**: it reintroduces a real bug
and asserts the suite goes red. A guard that passes with its bug reinstated is
decoration.

```bash
cd molplatte_preprocess && PYTHONPATH=$PWD/src python -m pytest tests -q
```

---

## Environment notes

Built and measured on 96 cores / 354 GB RAM / 2× RTX PRO 6000 Blackwell, conda env
`maven`, Python 3.12.13, torch 2.11 / cu130.

- **`faiss-gpu-cu12` 1.14.1 has no Blackwell (sm_120) kernels**, and its GPU
  index does not raise — it `abort()`s the process with CUDA error 209, so it
  cannot be caught and fallen back from. Retrieval therefore runs exact
  inner-product search through torch instead (`callbacks/exact_search.py`),
  which is the same arithmetic FAISS's `IndexFlatIP` performs. Measured on the
  real workload — 668,913 library rows, 20,000 queries, 300 dims, K=1000:
  **171.8 s on the FAISS CPU index against 0.4 s on the GPU, a 447× speedup with
  100% top-10 agreement.** Exact, not an ANN approximation. `index_type` still
  accepts `flat_cpu` and `hnsw` for reproducing older numbers.
- **ESM-2 650M** is downloaded on first use (2.5 GB, cached under
  `~/.cache/huggingface`).
- Pocket extraction needs **ProDy**; the notebook additionally needs
  **transformers**.

---

## Citation

```bibtex
@article{gim2024molpla,
  title   = {MolPLA: A Molecular Pretraining Framework for Learning Cores, R-Groups and Linkers},
  journal = {Bioinformatics},
  volume  = {40}, number = {Supplement 1}, pages = {i369--i380}, year = {2024},
  doi     = {10.1093/bioinformatics/btae256}
}
```
