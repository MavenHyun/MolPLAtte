# MolPLAtte — Design Notes (working draft)

> Status: **v1 built and validated end-to-end.** Corpora built, forward/backward
> verified, full Hydra training loop runs. Written from the MolPLA paper + released
> code, MolDAM, and MolDAM_prep, with every claim below measured on this machine.

MolPLAtte = **MolPLA's pretraining objectives** + **MolDAM's engineering** + **flavor chemistry**.

---

## 1. What MolPLAtte is

1. A variant of MolPLA with an **auxiliary pretraining objective for lead optimization
   (core decoration)**.
2. Pretrained on **flavor compounds** (FlavorDB) and **natural products** (COCONUT),
   not on GEOM/ZINC drug-like space.
3. The **condition vector** is either **protein-pocket context** or **neutral
   (ligand-only)** — replacing MolPLA's fixed 88-dim R-group functional-group vector.
4. Shares MolDAM's **anchored paradigm** (core + K R-groups, masked clone-joints,
   multi-decomposition enumeration per molecule).

Inheritance rule from the brief: `molplatte/` inherits **the primary traits of MolPLA,
not MolDAM** — MolDAM contributes code structure, implementation style, and the
efficient decomposition/corpus machinery.

---

## 2. MolPLA — the traits to inherit (verified against source)

Source: `/home/mogan/github/MolPLA` (`src/models/MolPLA.py`, `src/dataloaders/base.py`,
`src/utils.py`, `settings.yaml`).

### 2.1 The three views — paper notation vs code notation

The paper and the reference implementation use **different letters for the same objects**.
MolPLAtte standardises on the code's letters and records the mapping once:

| Code (`MolPLA.py`) | Paper (btae256) | Content |
|---|---|---|
| `G` | `𝒢_M` | the intact molecule graph, all linker flags cleared |
| `P` | `𝒢_{Q_{i,j,k}}` — the *query template* | core `𝒢_C` ∪ the R-groups **not** decoupled; one masked linker atom per *decoupled* R-group |
| `R` | `𝔾_{R_{i,j,k}}` | the decoupled R-groups, one graph each, each with one masked linker atom |
| `Q` | `𝒢_{D_{i,j,k}}` | `P ∪ R` — node-set union, pooled as one graph |

⚠️ The paper's `Q` is the code's `P`. Do not mix them. MolPLAtte uses **G / P / R / Q** as above.

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
§2, §5.1) and the one MolPLAtte must restore.

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

MolPLAtte generalises this slot to `{neutral, pocket}` — see §5.

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

### 3.1 Hard-won findings from MolDAM_prep that constrain MolPLAtte

1. **`naveja_recap` with a `ratio` threshold cannot produce multi-R-group
   decompositions** — 99.42% of instances have exactly one R-group at `ratio=2/3`, and
   even `ratio=1/2` only reaches 0.83%. Cause is structural: `find_putative_cores` takes
   RECAP *children* as cores, and a child is one connected fragment. A 5.9 h rebuild was
   spent discovering this.
   **Consequence for MolPLAtte:** MolPLA's whole formulation needs `k ≥ 2` R-groups for
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
MolPLAtte/
  molplatte_preprocess/         corpus builder   (mirrors MolDAM_prep)
    preprocess_flavor.py           the driver
    molplatte_prep/               the package
    scripts/build_all_corpora.sh
    docs/corpus_format.md
  molplatte/                    training repo    (mirrors MolDAM)
    docs/molplatte_design.md      this file
    src/{configs,data_modules,nnet_modules,loss_modules,lightning_modules,callbacks,scripts}
```

The brief names the preprocessing directory `molplatte_prep`; the pre-created
directory was `molplatte_preprocess`, so that is the repo name and
`molplatte_prep` is the Python package inside it.

---

## 5. Measured results

Current corpus statistics live in **[`molplatte_preprocess/docs/molplatte_corpus_eda.pdf`](../../molplatte_preprocess/docs/molplatte_corpus_eda.pdf)**,
regenerated with `molplatte_preprocess/generate_eda_report.py`.

Results from the macfrag / condition-vector era are archived in
[`archive/results_macfrag_condvec_era.md`](archive/results_macfrag_condvec_era.md).
They are superseded — the decomposition, the condition vector, the hash, the
scoring and the sampling unit have all changed since — but three findings from
that period still constrain the current design and are summarised at the top of
that file.

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
