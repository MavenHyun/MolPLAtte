# Pocket preprocessing guide

How structural pocket–ligand data becomes a MolPLAtte corpus, and which
decisions are load-bearing. Companion to `CORPORA.md`, which describes what is
on disk; this describes how it got there.

Everything here uses the **same decomposition** as the flavour corpora —
`naveja_recap`, `ratio=1/3`, `include_ring=True`, `max_cores=4`,
`min_rgroup_atoms=2`, `HASH_VERSION 4`. If that ever diverges, the R-group
hashes stop being comparable and retrieval breaks silently rather than loudly.

---

## 1. Sources

| | `~/datasets/crossdocked2020` | `~/datasets/tastepocket` |
|---|---|---|
| format | processed `pocket10` LMDB, 166,500 records | 343 mmCIF + 206 ligand SDF |
| pockets | 166,500 (one per record) | 343 T1 complexes |
| distinct ligands | 11,735 | 207 (T1), 186 validating |
| chemistry | drug targets | chemosensory receptors |
| on-target? | no | yes |
| big enough to train an encoder? | yes | no |

The split is deliberate: CrossDocked has the volume, tastepocket has the
relevance. Neither has both.

Raw structures are also on disk (`crossdocked_v1.1_rmsd1.0/**/*.pdb`) if you
need coordinates the LMDB does not carry.

---

## 2. Pipeline

```
source ──► ligand validation ──► decomposition ──► vocabulary ──► union library
                                                                       │
                                                              split metrics
                                                        (base_hit@K / novel_hit@K)
```

### 2.1 Reader

`molplatte_prep.readers.read_crossdocked` — registered as source `crossdocked`.

**One record per distinct ligand, not per pocket–ligand pair.** This is the
single most important choice in the reader. The LMDB has 166,500 records but
only 11,735 distinct ligands: CrossDocked is a *cross*-docking set, so the mean
ligand appears against 14.2 pockets and pyrophosphate against 556. Emitting
pairs would multiply every R-group's corpus count by that factor, unevenly, and
two things read those counts:

- the **frequency prior** every Hit@K is judged against, and
- the **logQ correction**, whose entire job is subtracting `log p(k)`.

The pocket side is not discarded. Each record carries `pocket_keys` — every LMDB
key whose pocket binds that ligand — so the pocket stage expands one ligand back
into its pairs without re-reading the LMDB.

**Split assignment is per ligand, with `test` taking precedence.** 86 ligands
bind both train and test pockets. Taking the first key's split — the obvious
implementation — places them in train and leaks the official test set through
ligand-level deduplication.

### 2.2 Ligand validation

`molplatte_prep.pocket_ligands`.

Structural datasets are full of molecules that are in the crystal but are not
binders. In CrossDocked they dominate the head: the twelve most frequent ligands
are ADP (1,903 pockets), SAH (1,121), MRD (640), OGA (582), ATP, NAD, GDP, UMP,
ANP, APC, AMP, PPV.

**Property filters do not separate them.** Measured on that head:

| ligand | QED | what it is |
|---|---:|---|
| S-adenosylhomocysteine | 0.35 | cofactor |
| benzamidine | 0.46 | crystallisation additive |
| staurosporine analogue | 0.30 | **real ligand** |
| estrone | 0.78 | **real ligand** |

Any QED threshold that removes the artifacts also removes genuine binders.

**Identity does separate them.** Every ligand has a PDB chemical component
(CCD) code, and artifacts are an enumerable set. CrossDocked has no CCD column,
but the code is embedded in `ligand_filename`
(`..._rec_1yfy_3ha_lig_...` → `3HA`) and parses for **100%** of records.

So the filter is: **CCD blocklist first, property screen second** for codes the
blocklist has not seen. Categories: `water`, `ion`, `cryo`, `buffer`, `glycan`,
`lipid`, `cofactor` (282 codes total).

Measured effect on CrossDocked:

```
distinct ligands  11,735 -> 11,268 kept   96.0%
pocket pairs     166,500 -> 147,648 kept  88.7%
```

4.0% of ligands but **11.3% of pairs** — the asymmetry is the point. Artifacts
are exactly the promiscuous head.

> **`MIN_HEAVY_ATOMS` is 5, not a drug-like 10.** Standard drug-likeness floors
> are calibrated on drugs. Odorants must be volatile to reach a receptor, so
> they are small by necessity. Propionate — 5 heavy atoms — is the cognate
> ligand of OR51E2 in `8F76`, the only genuine human olfactory receptor
> structure in the PDB. A drug-like floor discards exactly the chemistry this
> project exists to model. This bit us in testing; keep the floor at the corpus
> `SizeFilter` minimum.

> **Cofactors are a judgement call, not an error.** ATP in a kinase *is* the
> cognate ligand. They are excluded by default because they bind almost
> everything and would dominate the R-group vocabulary. `--keep-cofactors`
> re-admits them without loosening anything else.

### 2.3 Extracting ligands from structures

For structures rather than a prepared LMDB (tastepocket CIFs, raw CrossDocked
PDBs), use ProDy:

```python
from molplatte_prep.pocket_ligands import extract_ligands, pocket_residues, _parse_structure

st = _parse_structure(Path("…/8F76.cif"))
for key, verdict, sel in extract_ligands("…/8F76.cif"):
    if not verdict.ok:
        continue                       # verdict.reason says why
    pocket = pocket_residues(st, sel, cutoff=10.0)
```

`10.0` Å matches the `pocket10` convention CrossDocked's LMDB already uses, so
pockets extracted here are directly comparable to the ones in it.

> **Bond orders are not in a coordinate file.** RDKit infers them from geometry,
> and it is unreliable for exactly the aromatic and charged groups that matter.
> Observed: propionate in `8F76` comes back as `CCC(O)O` rather than
> `CCC(=O)[O-]`. Prefer the CCD ideal SDF
> (`structures/ligands/<CODE>_ideal.sdf`) whenever one exists — it carries
> deposited bond orders. Treat the geometry-inferred SMILES as a fallback.

### 2.4 Decomposition

Unchanged from the flavour corpora. Yields on these sources:

| source | decomposes |
|---|---:|
| CrossDocked ligands | 91.8% |
| tastepocket all-T1 | 84.1% |
| tastepocket taste-strict | 84.6% |

Losses are ordinary: 6–9% fail RECAP, ~8% fall outside the 5–50 heavy-atom band.

### 2.5 Vocabulary and the union library

**40% of CrossDocked's distinct R-groups are absent from
`coconut-flavordb-full`.** This matters more than it looks:
`RGroupLibraryRetrieval` masks targets whose hash is not in the library
(`rows_of() == -1`) and counts them as `n_target_missing`. Nothing crashes — the
queries are simply dropped from the metric. And the dropped ones are the rare,
novel tail, because matched R-groups have a median corpus count around 1,600.
Scoring against the pretraining library alone would report Hit@K over only the
chemistry the model already knew.

So build a union library and report the halves separately:

```bash
python scripts/build_union_vocab.py \
  ~/preprocessed/molplatte/union_vocab/base-full__crossdocked/rgroup_vocab.pkl.gz \
  ~/preprocessed/molplatte/coconut-flavordb-full/naveja_recap/rgroup_vocab.pkl.gz \
  ~/preprocessed/molplatte/crossdocked/naveja_recap/rgroup_vocab.pkl.gz
```

```
base 86,385 + novel 5,512 = 91,897 rows, effective 946
```

Counts are summed across inputs, because the frequency prior and logQ must
describe the library actually in use. Only the novel keys go into provenance
(`in_base` is `hash not in novel_hashes`) — 5.5k strings instead of 92k, and
`load_vocabulary` copies provenance wholesale so the flag reaches training
without touching `RGroupEntry`.

Verified on 11,950 real CrossDocked targets: **0 missing** against the union
versus ~19% dropped against base alone; 21.0% novel, 79.0% base.

---

## 3. Running it

```bash
cd molplatte_preprocess
export PYTHONPATH=$PWD/src
A=~/preprocessed/molplatte/annotations
D=~/preprocessed/molplatte

python preprocess_flavor.py \
  --source crossdocked --method naveja_recap --output-path $D/crossdocked.new \
  --workers 96 --batch-size 100 --layout hash3 --progress-every 4000 \
  --core-ratio 0.3333333333333333 --max-cores 4 --max-rgroups 8 \
  --min-rgroup-atoms 2 --keep-stereo --no-neutralise \
  --min-heavy-atoms 5 --max-heavy-atoms 50 \
  --condvec-mode flavor \
  --flavor-measured $A/flavor_measured.jsonl \
  --flavor-mined    $A/flavor_documented.jsonl
```

> `--output-path` writes records at the path given, **without** a
> `<method>/` level. The rest of the tooling expects
> `<corpus>/naveja_recap/`, so restructure before building the vocabulary:
> ```bash
> mkdir -p $D/crossdocked/naveja_recap && mv $D/crossdocked.new/* $D/crossdocked/naveja_recap/
> ```

Then vocabulary, union, and the manifest:

```bash
python enumerate_rgroups.py --corpus $D/crossdocked/naveja_recap --workers 96
python scripts/build_union_vocab.py <out> <base-vocab> <crossdocked-vocab>
python scripts/write_corpora_manifest.py
```

Train against the union library:

```bash
python run.py … \
  data_module_kwargs.dataset_version=crossdocked \
  rgroup_library.vocab_path=$D/union_vocab/base-full__crossdocked/rgroup_vocab.pkl.gz
```

Relevant flags: `--keep-artifact-ligands` (disable filtering entirely),
`--keep-cofactors` (re-admit cofactors only).

### 3.1 tastepocket

```bash
D=~/preprocessed/molplatte/tastepocket

python scripts/extract_tastepocket_ligands.py --out $D/ligands.jsonl
python scripts/extract_tastepocket_pockets.py --ligands $D/ligands.jsonl \
                                              --out $D/pockets.jsonl
CUDA_VISIBLE_DEVICES=1 python scripts/embed_pockets_esm.py \
    --pockets $D/pockets.jsonl --out $D/pocket_esm2_650M.npz
# flavour labels: tables, then a molecule-only LLM pass -> $D/flavor_labels.jsonl
python scripts/build_tastepocket_dataset.py \
    --pockets $D/pockets.jsonl --embeddings $D/pocket_esm2_650M.npz \
    --flavor $D/flavor_labels.jsonl --ligands $D/ligands.jsonl \
    --out $D/dataset.jsonl \
    --folds-out ~/preprocessed/molplatte/tastepocket_corpus/naveja_recap/folds.json
```

Then decompose. **`--no-dedup` is required**: the same ligand at a different
receptor is a different training example, and dedup-by-washed-structure collapses
269 records to 164.

```bash
python preprocess_flavor.py \
  --source tastepocket --method naveja_recap --no-dedup \
  --output-path $DEST --workers 16 --batch-size 20 --layout hash3 \
  --core-ratio 0.3333333333333333 --max-cores 4 --max-rgroups 8 \
  --min-rgroup-atoms 2 --keep-stereo --no-neutralise \
  --min-heavy-atoms 5 --max-heavy-atoms 50 \
  --condvec-mode two_part --pocket-source stored --pocket-dim 1280 \
  --flavor-measured $A/flavor_measured.jsonl \
  --flavor-mined    $A/flavor_documented.jsonl
```

Training runs then go through `src/scripts/run_step2_pocket_cv.sh`, which expands
the flavour-only STEP 1 checkpoint (324 → 356 query inputs, new columns zeroed)
and warm-starts every fold from it.

---

## 4. Reading the results

| metric | means |
|---|---|
| `library/hit@K` | micro average over all queries |
| `library/base_hit@K` | queries whose R-group was in the pretraining vocabulary |
| `library/novel_hit@K` | queries whose R-group was **not** |
| `library/n_target_missing` | queries dropped entirely — should be 0 with a union library |

`novel_hit@K` is the one that answers whether pocket conditioning generalises.
A healthy `hit@K` with a collapsed `novel_hit@K` means the model is retrieving
familiar chemistry, which the pocket stage was not built to demonstrate.

**CrossDocked's effective vocabulary is 1,478 against `coconut-flavordb-full`'s
889**, despite 35× fewer records — its R-groups are more evenly spread, so
retrieval here is *harder*. Numbers are not comparable across the two corpora in
either direction. Give pocket-stage runs their own baseline.

---

## 5. The tastepocket finetuning set

Built by `extract_tastepocket_ligands.py` → `extract_tastepocket_pockets.py` →
`embed_pockets_esm.py` → `build_tastepocket_dataset.py`, then decomposed with
`--source tastepocket`.

### 5.1 Ligands

186 of 207 distinct T1 cognate ligands validate. Getting there needed a fix to
the CCD blocklist: calibrated on CrossDocked's enzymes and drug receptors, it
rejected `GLU`, `SPM`, `SPD` and `PUT` as *buffer*. At a T1R, CaSR or TAAR those
are the stimulus — glutamate is umami, spermine is the kokumi agonist, putrescine
is the canonical TAAR13c odorant. Four codes over nine complexes, and among the
most on-target entries in the set. `CHEMOSENSORY_COGNATE` re-admits them per
code; the strict default is unchanged, so CrossDocked is unaffected.

> **`TLA` stays blocked.** L-tartrate is the thaumatin *crystallant* — 147
> entries carry it and nothing else. Rescuing by category rather than by code
> would have admitted it.

### 5.2 Ligand identity is structural, not a flag

A ligand named `TRP` is not the 14 tryptophans in the protein backbone. mmCIF
records free amino acids as `ATOM`, not `HETATM`, so `hetero` cannot find them —
and a bare `resname TRP` match sweeps up the whole polymer. In `7DTU` that turned
2 real ligands into 30, and 445 across the set, every extra one a pocket built
around a backbone residue. Nothing errored: the pockets were the right size and
full of real atoms.

`ligand_instances()` keeps a copy if it sits outside a polymer chain (≥20 amino
acids) **or** is flagged hetero. The first arm catches free amino-acid agonists —
in `7DTU` the tryptophan is chain G, one residue, 15 atoms because it carries
OXT. CaSR and T1R are amino-acid sensors, so this is the common case here.

Fixing it took the site count from 2,269 to **1,255**.

### 5.3 Flavour labels

Tables first (`measured` → `mined`), then a frontier LLM for the rest, then
`unknown`. **The prompt sees the molecule only** — name and SMILES, never the
receptor. Telling the model "this binds TRPM8" would make it answer *cooling*,
and the flavour half of the condvec would become a re-encoding of the pocket
half: conditioning would then show a large lift meaning nothing, because both
halves would carry the same variable.

| source | n |
|---|---:|
| measured | 92 |
| llm | 10 |
| mined | 8 |
| llm_abstain / unlabelled | 159 |

41% carry a real label. The rest is not a preprocessing failure: of 118 molecules
the tables could not resolve, 92 are modulators, lipids or cofactors with no
percept at all, and 19 more are **insect pheromones** — bombykol, bombykal,
honeybee queen mandibular pheromone. Genuine stimuli with no human descriptor.
91 of 343 T1 entries are insect OBPs, so a large part of this set is unlabellable
in a 22-term human vocabulary by construction.

Blind control: 63 ligands whose labels the measured table already knows were
graded without telling the grader which they were. Precision 0.73, recall 0.39 —
and all three *confident* disagreements were the table being wrong (capsaicin
labelled `herbal, odorless`; glutamate `dairy, odorless, roasted`). That thread
led to a real defect: 145 rows of `flavor_measured.jsonl` assert `odorless`
alongside an odour class. Fixed at the encoder; `CONDVEC_VERSION` → 3.

### 5.4 Grain and folds

**One record per (ligand, receptor)** — 1,255 sites → **269 records**. Tryptophan
at four genuinely different receptors is four datapoints; 160 crystallographic
copies of alanine in one structure is one. The instance grain inflates the
R-group frequency prior unevenly, and that prior is both what Hit@K is judged
against and what logQ subtracts, so inflating it moves the number being optimised.

**Five folds, not one split.** A 15% holdout is ~40 records; the seed would move
the answer more than the model does. The deliverable checkpoint trains on all 269.

**Folds cut whole connected components** of the (ligand, receptor) bipartite
graph, so ligand- and receptor-disjointness both hold by construction and nothing
is dropped. Splitting by receptor and patching ligand conflicts afterwards cost 64
of 269 records and still left the halves unbalanced. 45 components, largest 20.4%.

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| records | 56 | 58 | 52 | 51 | 52 |

Verified at the dataloader level, not just in the manifest: every fold disjoint on
molecules, ligands **and** receptors.

> **Fold 1 asks a harder question.** TRPV1, CaSR, TRPA1, OTOP1 and CSP are each a
> single component, so they are all-or-nothing. Fold 1 is 52/58 TRPV1+TRPA1, which
> makes it a held-out receptor *family* rather than a held-out receptor. Read it
> separately rather than averaging it in.

Why none of this can be random: pooled ESM-2 pocket embeddings identify the
receptor family with **98.5% 1-NN accuracy across different PDB entries** (27.4%
majority baseline). A random split puts the same protein on both sides and
reports memorisation as generalisation.

### 5.5 The pocket encoder

Frozen ESM-2 650M, residues within 10 Å, mean-pooled over every pocket residue of
every chain at once — pooling per chain and averaging would give a 5-residue
contact the same weight as a 40-residue wall. The embedding is 1D; the *selection*
is 3D, which is what keeps it usable for GPCRs where pocket residues are far apart
in sequence (in `8F76` the propionate site draws on 100–108 and 151–157).

> **The BOS offset is the trap.** ESM prepends `<cls>`, so residue *i* is token
> *i+1*. Forgetting it shifts every pocket by one residue and raises nothing.
> `verify_offset()` asserts the alignment against a probe instead of trusting a
> comment.

The vector is stored **raw**, at 1280. `StoredPocketCondVec` validates width and
finiteness and hands it through unreduced; `PocketConditioning` learns the
reduction inside the model, where gradients decide what to keep. Storing it
pre-reduced would freeze that choice at preprocessing time; feeding it raw to the
query would make it 80% of the input and let it dominate by magnitude.

> **uint8 storage destroyed it once.** The corpus stored `rgroup_condvecs` as
> `np.uint8` — correct for 24 binary bits, catastrophic for negative floats:
> −6.668 wraps to 250, everything in (−1, 1) truncates to 0. Right shape, right
> dtype, none of the information. Encoders now declare `storage_dtype` and a
> non-round-tripping integer cast raises.

### 5.6 Corpus

```
records 243   decompositions 734   R-groups 906   condvec 1304 = 24 + 1280
```

Dedup is **off** for this source: the same ligand at a different receptor is a
different training example, and dedup-by-washed-structure collapsed 269 records to
164, discarding exactly the pocket variation this stage learns.

42 of 263 tastepocket R-groups (16.0%) are absent from the pretraining vocabulary,
38 also absent from CrossDocked. The three-way union library is 91,935 rows and
covers every one, so `n_target_missing` is 0 rather than silently dropping the
novel tail from the metric.

---

## 6. Known gaps

- The CCD blocklist (282 codes) is a curated subset, not exhaustive. Unlisted
  codes fall through to the property screen rather than being trusted.
- The build tag (`r333-m2-h4-flavor24v2`) encodes decomposition and condvec, but
  **not which library a run scored against**. Two runs on the same corpus with
  different libraries still tag identically.
- CrossDocked has no pocket embeddings yet, so the pocket encoder currently
  trains on 269 tastepocket records. That is far too few to learn a pocket
  representation from scratch; CrossDocked's 11,268 ligands over 147,648 pairs
  are the only substrate with enough receptor variety.

  The path is available: full receptors sit at
  `crossdocked_v1.1_rmsd1.0/<target>/*_rec.pdb` across 2,474 target directories.
  Use those, NOT the prepared `*_pocket10.pdb` files -- the corpus recipe embeds
  the whole chain and then selects pocket residues from it, so embedding a
  pre-cut pocket would strip the protein context ESM depends on and put the two
  corpora in different spaces while every shape still matched.
- Geometry is not used. The PLM path was chosen first because it needs no change
  to the data pipeline; whether 3D structure buys anything over sequence is an
  open question that an EGNN ablation would answer.
