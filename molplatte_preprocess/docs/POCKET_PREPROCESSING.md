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
| distinct ligands | 11,735 | 207 (T1), 26 taste-strict |
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

## 5. Known gaps

- The CCD blocklist (282 codes) is a curated subset, not exhaustive. Unlisted
  codes fall through to the property screen rather than being trusted.
- The build tag (`r333-m2-h4-flavor24v2`) encodes decomposition and condvec, but
  **not which library a run scored against**. Two runs on the same corpus with
  different libraries currently tag identically.
- `PocketCondVec` still returns zeros — the pocket encoder does not exist. Every
  corpus here is substrate for that stage, not the stage itself.
