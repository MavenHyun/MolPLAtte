# molpallete_preprocess

Corpus builder for [MolPallete](../docs/molpallete_design.md): flavor compounds and
natural products → anchored **core + k R-groups** records for MolPLA-style
masked-linker pretraining.

MolPallete = MolPLA's pretraining objectives + MolDAM's engineering + flavor
chemistry. This repo is the *preprocessing* half: it reads FlavorDB / COCONUT
structures, decomposes each molecule into every anchored core it admits, and
writes one `.pt` per molecule holding the intact graph plus the bookkeeping for
each decomposition. Sampling of the `(decomposition, islinked)` training pair
happens in the training repo, not here — see
[`docs/corpus_format.md`](docs/corpus_format.md) for why.

---

## Source corpora

| source | records | after dedup | keyed by | what it carries |
|---|---:|---:|---|---|
| **FlavorDB** (`~/datasets/flavordb`) | 25,595 | 25,595 | PubChem CID | isomeric SMILES + descriptors (`properties.csv`), semicolon-delimited `flavor_profile` labels (`flavordb_molecules.csv`, 98.1% populated, 715 distinct tokens) |
| **COCONUT** (`coconut_sdf_3d-08-2026.sdf`, 2.3 GB) | 737,343 | **489,395** | `CNP…` | 3D molblocks and nothing else |

Two things about these sources drive the reader code
(`molpallete_prep/readers.py`):

* **FlavorDB is read from the CSVs, not the SDFs.** `flavordb_3d.sdf` carries no
  SMILES at all and covers only 70.3% of the set. `properties.csv` has isomeric
  SMILES for all 25,595, and its `HeavyAtomCount` column lets `read_flavordb`
  apply the size filter *before* parsing a SMILES it is going to reject anyway.
* **COCONUT's only SDF tag is `IDENTIFIER`.** No SMILES, no InChIKey, no
  taxonomy. Chemistry is therefore recomputed from each 3D molblock with
  `Chem.MolToSmiles`. Identifiers look like `CNP0252853.1`, where the suffix
  indexes conformer/stereo variants of the same compound; the 737,343 records
  collapse to **489,395 distinct base IDs**. `read_coconut(dedup_variants=True)`
  is the default because keeping the variants would triple-count ~250K compounds
  in the R-group statistics. The flag stays because a stereo-aware study might
  legitimately want them.

Both readers stream (`ForwardSDMolSupplier` / `csv.DictReader`) and never hold a
whole file in memory.

**Size filtering.** COCONUT skews large: median 31 heavy atoms, p90 63, max 475,
against FlavorDB's median 23 / p90 45. Macrolides, polysaccharides and large
glycosides at the top of that range are not flavor-relevant and would dominate
the R-group vocabulary. `SizeFilter` defaults to `[5, 50]` heavy atoms — roughly
FlavorDB's 93rd percentile — which drops the largest 4.4% (~32K) of COCONUT.
`--max-heavy-atoms 40` gives a more volatiles-focused corpus (87% of COCONUT);
the flag exists precisely because that trade-off is a judgement call.

---

## Repo layout

```
molpallete_preprocess/
├── preprocess_flavor.py           CLI driver: source → per-mol .pt corpus
├── scripts/build_all_corpora.sh   batch build of every (source, method) corpus
├── molpallete_prep/
│   ├── __init__.py                public surface (re-exports the whole API)
│   ├── readers.py                 FlavorDB / COCONUT streaming readers, SizeFilter
│   ├── decompose.py               wash(), Naveja putative cores, ring-aware fallback,
│   │                              Decomposition / RGroupInfo dataclasses
│   ├── decomposers/
│   │   ├── __init__.py            registry: list_methods / family_of /
│   │   │                          get_anchored_decomposer / get_fragment_decomposer
│   │   ├── naveja_recap.py        RECAP children + Naveja core-size filter  (anchored)
│   │   ├── bemis_murcko.py        Murcko scaffold core + side chains        (anchored)
│   │   ├── macfrag.py             MacFrag cleavable bonds → partitions      (fragment)
│   │   ├── synton.py              Synt-On retrosynthetic bonds → partitions (fragment)
│   │   ├── anchored_common.py     core-atoms → Decomposition, safety filters
│   │   └── fragment_common.py     cleavable bonds → FragmentPartition
│   ├── anchored_from_partition.py the bridge: flat partition → anchored stars
│   ├── molpla_instance.py         MolPlaInstance, build_instance, sample/enumerate
│   │                              islinked, decomposition_record
│   ├── condvec.py                 NeutralCondVec (97-dim) / PocketCondVec (stub)
│   ├── mol_features.py            RDKit Mol ↔ PyG with MolPLA's exact feature schema
│   ├── graph_ops.py               detach_rgroups_multi / attach_rgroups (anchored)
│   ├── fragment_types.py          FragmentPartition / FragmentInfo / CutBond
│   ├── fragment_graph_ops.py      detach_fragments / attach_fragments (flat paradigm)
│   ├── graph_hash.py              Weisfeiler-Lehman subgraph hash (vocabulary key)
│   ├── vocab.py                   vocabulary build/merge/save/load
│   ├── lmdb_store.py              LMDB corpus reader/writer + dehydrate/hydrate
│   ├── preprocess/writer.py       per-mol .pt layout, atomic __meta__/__manifest__
│   └── vendor/                    vendored MacFrag and Synt-On — read-only
├── docs/corpus_format.md          on-disk record format + verification recipe
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

`vocab.py` and `lmdb_store.py` come over from MolDAM_prep intact. `lmdb_store`'s
`dehydrate` is on the hot path (`preprocess/writer.py` calls it on every record);
the LMDB store itself and the vocabulary builder are **not yet wired into the
driver** — they are available for a vocabulary/co-occurrence pass that has not
been run for MolPallete.

---

## Decomposition methods

MolPallete needs every molecule expressed in MolPLA's **anchored star topology**:
one core plus *k* R-groups, each joined to the core by exactly one shared linker
atom. Two families feed that shape, and the registry
(`molpallete_prep.decomposers`) knows which is which via `family_of(method)`.

**Native anchored** — return `List[Decomposition]` directly:

| `--method` | what it does |
|---|---|
| `naveja_recap` | RDKit RECAP children kept when they hold ≥ `ratio · nHA(M)` (Naveja 2019 core-size filter, default 2/3), plus a ring-aware fallback over non-ring single bonds. **MolPLA's own method.** |
| `bemis_murcko` | One decomposition per molecule: core = Murcko scaffold, R-groups = the side chains hanging off it. |

**Multi-cut fragment** — return `List[FragmentPartition]`, re-framed by
`anchored_from_partition.partitions_to_decompositions`:

| `--method` | what it does |
|---|---|
| `macfrag` | MacFrag (Diao 2023): BRICS-like environments + igraph block merging. **The default.** |
| `synton` | Synt-On: curated reaction-based SMARTS disconnections. |

### Measured comparison — 400 FlavorDB molecules, heavy atoms in [5, 50]

| method | no-decomp | dec/mol | R-groups/dec | k ≥ 2 | throughput |
|---|---:|---:|---:|---:|---:|
| `naveja_recap` (ratio 2/3) | 0.2% | 10.65 | **1.04** | **3.70%** | 678 mol/s |
| `naveja_recap` (ratio 1/2) | 0.2% | 12.81 | **1.06** | **5.54%** | 634 mol/s |
| `bemis_murcko` | 18.2% | 1.00 | 4.94 | 77.37% | 9749 mol/s |
| `macfrag` (raw partitions) | 12.2% | 1.00 partition | 7.52 frags | — | 235 mol/s |
| `synton` (raw partitions) | 50.2% | 1.00 partition | 3.36 frags | — | 7.2 mol/s |

The two fragment rows are *raw partitions*, so their "R-groups" column is
fragments per partition and `k ≥ 2` does not apply — a flat partition has no
core. Re-framing MacFrag's partitions through `partitions_to_decompositions`
gives the numbers that actually matter:

| ratio | no-decomp | cores/mol | R-groups/core | k ≥ 2 | core nHA |
|---:|---:|---:|---:|---:|---:|
| 0.33 | 12.2% | 79.30 | 3.26 | 89.67% | 20.7 |
| **0.50** | 12.2% | **64.45** | **3.34** | **89.71%** | **22.5** |
| 0.67 | 16.8% | 45.65 | 3.23 | 88.61% | 24.7 |

For reference, MolPLA reports **~4.04 cores per molecule** and **~20.8-heavy-atom
cores** on GEOM. So `--core-ratio 0.5` matches MolPLA's core *size* while
producing far more candidate cores than are wanted; `--max-cores` prunes the
surplus (`partitions_to_decompositions` deduplicates on `core_atoms`, then keeps
the best `max_cores` by *more R-groups first, larger core second*). MolPLA itself
dropped molecules above 10 cores, its 99th percentile being 11.

### The headline finding

**`naveja_recap` — MolPLA's own method — collapses to ~1 R-group per
decomposition on flavor chemistry.** At ratio 2/3 the mean is 1.04 R-groups and
only **3.70%** of decompositions reach k ≥ 2; loosening to ratio 1/2 buys 1.06
and 5.54%. At k = 1 the `islinked` subset space is `2¹ − 1 = 1` — there is
exactly one training instance per decomposition, the "subset of R-groups" is
always the whole set, and MolPLA's core-decoration objective has nothing to
decorate. Both are degenerate.

The cause is structural, not a tuning problem: `find_putative_cores` takes RECAP
*children* as cores, and a RECAP child is one connected fragment, so the core
boundary almost always has a single crossing. This reproduces exactly what
MolDAM_prep measured on ZINC — **99.42% single-R-group at ratio 2/3**, and even
ratio 1/2 only reaching 0.83% multi-R-group — after a 5.9-hour rebuild spent
discovering it. Two independent chemical spaces, same failure.

Hence the default is `macfrag` re-framed through the fragment tree: 3.34
R-groups per core with **89.71% at k ≥ 2**, at a core size that matches what
MolPLA trained on. `naveja_recap` stays in the registry as the paper-faithful
baseline, not as a recommendation.

`bemis_murcko` is the interesting near-miss: 4.94 R-groups per decomposition and
77.37% at k ≥ 2 at 9,749 mol/s, but it yields exactly one core per molecule and
nothing at all for 18.2% of the set (acyclic molecules have no scaffold — and
flavor space is full of acyclic terpenoids, esters and fatty aldehydes). One core
per molecule kills MolPLA's "single molecule → multiple putative cores" property,
which is what makes the graph-contrastive loss see the same molecule through
different decompositions.

### `synton` caveat

`synton` yields **no partition for 50.2%** of flavor molecules, against 21% on
ZINC — its rule set is tuned for synthesizable drug-like space, and half of
flavor chemistry does not look like that. It also runs at **7.2 mol/s**
single-core, ~30× slower than the next slowest method and ~1,350× slower than
`bemis_murcko`. It is usable (489K COCONUT molecules would take ~19 core-hours,
trivially parallel across 96 cores) but there is no reason to prefer it here.

All methods run through a safety filter that drops ring-crossing cuts. The
anchored path additionally rejects **geminal joints** (two R-groups sharing one
core linker atom): `graph_ops.detach_rgroups_multi` represents at most one joint
per template atom, and MolPLA's `is_linker` marker is likewise one bit per atom.

---

## Condition vector

MolPLA conditions R-group retrieval on a binary functional-group vector of the
*target* R-group. The paper says `c_R ∈ [0,1]^87`; the released code allocates 88
and fills it from `thermo.functional_groups`. `thermo` is not a dependency here,
and its 88 checks are bulk-thermodynamic classes rather than flavor chemistry, so
`molpallete_prep/condvec.py` defines its own:

* `--condvec-mode neutral` → **`NeutralCondVec`, 97 binary bits**: 85 RDKit
  `Chem.Fragments.fr_*` counters plus 12 SMARTS patterns central to flavor and
  aroma chemistry that `fr_*` does not cover (pyrazine, pyrrole, pyranone,
  alkylthiophene, di/trisulfide, thioester, acetal, ketal, isoprene unit,
  cyclohexene, internal olefin). Binary, not count-valued — MolPLA's vector is a
  presence indicator, and counts would let one R-group's magnitude dominate the
  concatenated query. This is the working default.
* `--condvec-mode pocket` → **`PocketCondVec`, a declared interface with an
  unfed implementation.** FlavorDB and COCONUT carry no protein pairings, so it
  returns zeros. That is deliberate: all-zero is exactly the paper's `Cond. None`
  ablation, which drops retrieval MRR from 0.2616 to 0.0056, so a pocket run that
  is silently unfed shows up as collapsed retrieval rather than as
  plausible-looking noise. (`Cond. All`, all-ones, drops it below 0.0001 — a
  degenerate condition is worse than no condition.) Wiring a taste-receptor or
  CrossDocked pocket encoder into `PocketCondVec._encoder` is the only change
  needed on this side.

---

## Installation

```bash
conda activate maven            # or: conda create -n maven python=3.12 -y
pip install -e .                # editable, source of truth
# or:
pip install -r requirements.txt
```

Import check:

```bash
python -c "from molpallete_prep import list_methods, build_instance; print(list_methods())"
# ['bemis_murcko', 'macfrag', 'naveja_recap', 'synton']
```

`python-igraph` is a hard dependency rather than an extra because `macfrag` is
the default method; without it the driver fails at the first molecule.

---

## Usage

The driver is `preprocess_flavor.py` at the repo root.

### FlavorDB (fast — the whole set fits in minutes)

```bash
python preprocess_flavor.py \
    --source flavordb \
    --source-path ~/datasets/flavordb \
    --output-path ~/preprocessed/molpallete/flavordb/macfrag \
    --method macfrag \
    --core-ratio 0.5 --max-cores 10 --max-rgroups 8 \
    --min-heavy-atoms 5 --max-heavy-atoms 50 \
    --keep-stereo \
    --condvec-mode neutral \
    --workers 32 --batch-size 200 \
    --layout hash3 \
    --progress-every 1000
```

### COCONUT (489K molecules after dedup)

```bash
python preprocess_flavor.py \
    --source coconut \
    --source-path ~/datasets/coconut/coconut_sdf_3d-08-2026.sdf \
    --output-path ~/preprocessed/molpallete/coconut/macfrag \
    --method macfrag \
    --core-ratio 0.5 --max-cores 10 --max-rgroups 8 \
    --min-heavy-atoms 5 --max-heavy-atoms 50 \
    --keep-stereo \
    --condvec-mode neutral \
    --workers 64 --batch-size 200 \
    --layout hash3 \
    --progress-every 5000 \
    --resume
```

### CLI surface

| flag | default | notes |
|---|---|---|
| `--source {flavordb,coconut}` | *required* | picks the reader; `--source-path` overrides the default location in `readers.SOURCES`. |
| `--source-path` | reader default | `~/datasets/flavordb` or `~/datasets/coconut/coconut_sdf_3d-08-2026.sdf`. |
| `--method` | `macfrag` | any of `list_methods()`. `family_of()` decides whether the output goes through `partitions_to_decompositions`. |
| `--core-ratio` | `0.5` | fragment family: a candidate core must hold ≥ `ratio · n_atoms`. Anchored family: this *is* `naveja_recap`'s Naveja core-size `ratio`. Same flag, two meanings — both recorded in `__meta__.json`. |
| `--max-cores` | `10` | cores kept per molecule after dedup + ranking. Mirrors MolPLA's own cutoff. |
| `--max-rgroups` | `8` | rejects cores above this k. The training-time subset space is `2^k − 1`, so this bounds sampling cost, not corpus size. |
| `--no-ring-aware` | off | `naveja_recap` only: disables the non-ring-single-bond fallback (`include_ring=False`). |
| `--min-heavy-atoms` / `--max-heavy-atoms` | `5` / `50` | `SizeFilter`, applied before the expensive decomposition step. |
| `--keep-stereo` / `--no-keep-stereo` | `--keep-stereo` | maps to `wash(remove_stereo=...)`. See "Differences from MolDAM_prep". |
| `--condvec-mode {neutral,pocket}` | `neutral` | `neutral` = the 97-bit flavor vector; `pocket` currently emits zeros. |
| `--workers` | `cpu_count() − 2` | worker pool. Raising above ~30 helps only if `--batch-size` goes up with it. |
| `--batch-size` | `200` | molecules per task handed to a worker. |
| `--layout {flat,shard2,shard3,hash3}` | `hash3` | directory bucketing; **keep `hash3`** for these sources. |
| `--limit-mols` / `--sample-fraction` / `--sample-seed` | — / — / `42` | truncation and seeded Bernoulli subsampling for quick iteration. |
| `--progress-every` | `5000` | molecules between progress lines. |
| `--resume` | off | rebuild the index from the prior `__meta__.json` and skip molecules already on disk (`writer.existing_ids`). |
| `--overwrite` | off | **deletes** the output directory first. Without `--resume` or `--overwrite` the driver refuses to write into a non-empty destination. |

### Verifying a corpus

After a build, check the index before shipping it anywhere:

```bash
python - <<'PY'
import json, random, torch
from pathlib import Path
from molpallete_prep.preprocess import path_for

root = Path("~/preprocessed/molpallete/flavordb/macfrag").expanduser()
meta = json.loads((root / "__meta__.json").read_text())

n = meta["n_records"]
assert len(meta["ids"]) == n == len(meta["n_decomps"]) == len(meta["n_rgroups_per_decomp"])
print(f"{n} records, layout={meta['layout']}, method={meta['method']}, "
      f"condvec={meta['condvec_mode']}/{meta['condvec_dim']}")

missing = [i for i in meta["ids"] if not path_for(root, i, meta["layout"]).is_file()]
assert not missing, f"{len(missing)} indexed records missing on disk"

for mol_id in random.sample(meta["ids"], min(50, n)):
    rec = torch.load(path_for(root, mol_id, meta["layout"]), weights_only=False)
    assert rec["mol_id"] == mol_id
    assert rec["n_decomps"] == len(rec["decompositions"])
    n_atoms = rec["original"]["num_nodes"]
    for dec in rec["decompositions"]:
        assert dec["n_rgroups"] == len(dec["rgroups"]) == len(dec["rgroup_hashes"])
        assert dec["rgroup_condvecs"].shape == (dec["n_rgroups"], meta["condvec_dim"])
        assert max(dec["core_atoms"]) < n_atoms
print("ok")
PY
```

The fuller recipe — including the round-trip check you should re-run after
bumping RDKit — is in [`docs/corpus_format.md`](docs/corpus_format.md).

---

## Environment (measured)

96 cores · 354 GB RAM · 2× RTX PRO 6000 Blackwell (97 GB each) · 1.0 TB free on
`/`. Conda env `maven`, Python 3.12.13. Present: rdkit 2026.03.2, torch
2.11.0+cu130, torch_geometric 2.7.0, lightning 2.6.1, networkx, lmdb, zstandard,
faiss-gpu-cu12 1.14.1, pyarrow. **Not installed: `thermo`, `faiss-cpu`** —
`thermo` is why `condvec.py` reimplements the functional-group vector from RDKit
rather than importing MolPLA's.

All throughput figures in this README are **single-core** (400 FlavorDB
molecules, heavy atoms in [5, 50]), so they scale roughly linearly with
`--workers`.

---

## Differences from MolDAM_prep

This repo is adapted from [MolDAM_prep](https://github.com/MavenHyun/MolDAM_prep).
Five things changed, each for a measured reason:

1. **Stereochemistry is preserved by default.** MolDAM_prep calls
   `wash(remove_stereo=True)`; MolPallete defaults to `--keep-stereo`
   (`wash(remove_stereo=False)`). Cis/trans isomerism is chemically load-bearing
   in flavor: **(Z)-3-hexen-1-ol is "cut grass"/green, (E)-3-hexen-1-ol is not** —
   same graph, same formula, different odour. Stripping stereo would merge them
   into one R-group vocabulary entry and make the retrieval target ambiguous. The
   cost is a larger vocabulary and a `chiral_tag` / `bond_stereo` feature channel
   that now actually varies; `--no-keep-stereo` restores MolDAM's behaviour for
   comparability runs.
2. **`mol_id` replaces `zinc_id`.** Ids are `FDB{cid}` for FlavorDB and the
   COCONUT base identifier (`CNP0252853`, suffix stripped) for COCONUT. The
   record and both JSON sidecars use `mol_id` throughout.
3. **Metadata writes are atomic.** `preprocess/writer.atomic_write_json` writes to
   a temp file in the destination directory, fsyncs, then `os.replace` (atomic on
   POSIX). `__meta__.json` runs to tens of MB on a large corpus and a torn write
   destroys the whole index — MolDAM_prep's handoff flagged this as its clearest
   unfixed defect.
4. **Resume is supported.** `writer.existing_ids` returns the subset of ids
   already on disk, so `--resume` restarts a killed COCONUT build without
   redoing finished work. MolDAM_prep had no resume; an interrupted run started
   over.
5. **A `hash3` shard layout exists.** MolDAM_prep buckets on the last N
   characters of the id, which assumes fixed-width, uniformly-distributed ids —
   true of 16-digit ZINC ids, false of `FDB4` / `FDB25595` / `CNP0252853`.
   Suffix-slicing those buckets them very unevenly (and `FDB4` is shorter than
   the slice). `hash3` buckets on a 2-byte blake2b digest of the `mol_id`
   instead, which is uniform by construction. `flat` / `shard2` / `shard3` are
   still there for MolDAM-compatible corpora.

The provenance block also moved *inside* the corpus (`__meta__.json`) rather than
living only in a run log: MolDAM_prep lost the run manifest for seven corpora
during a directory move and had to reconstruct decomposition ratios from file
mtimes.

---

## Status

Built and tested: readers, all four decomposers, the partition→anchored bridge,
`build_instance`, the neutral condition vector, the per-mol `.pt` writer.

Not yet built:
* `PocketCondVec` returns zeros — no pocket source is wired (see above).
* `core_smiles` is left `""` for the fragment family; computing it costs a SMILES
  round-trip per candidate core and nothing downstream reads it.
* `vocab.py` / `lmdb_store.py` are carried over intact but no MolPallete
  vocabulary or co-occurrence pass has been run.
* No FlavorDB or COCONUT corpus has been built at full scale yet; every number in
  this README comes from the 400-molecule FlavorDB benchmark.

---

## Citations

* Gim, M. *et al.* **MolPLA: a molecular pretraining framework for learning cores,
  R-groups and their linker joints.** *Bioinformatics* **40**(Suppl 1),
  i369–i380 (2024). doi:[10.1093/bioinformatics/btae256](https://doi.org/10.1093/bioinformatics/btae256).
  Reference implementation: [github.com/dmis-lab/MolPLA](https://github.com/dmis-lab/MolPLA).
* Diao, Y. *et al.* **MacFrag: segmenting large-scale molecules to obtain diverse
  fragments with high qualities.** *Bioinformatics* **39**(1), btad012 (2023).
  Vendored at `molpallete_prep/vendor/macfrag.py`; original:
  [github.com/yydiao1025/MacFrag](https://github.com/yydiao1025/MacFrag).
* Chuiko, Y. *et al.* **Synt-On:** retrosynthetic SMARTS-rule fragmentation of
  synthesizable chemical space. Vendored at
  `molpallete_prep/vendor/synton/`.
* Naveja, J. J. *et al.* **A general approach for retrosynthetic molecular core
  analysis.** *J. Cheminform.* **11**, 61–69 (2019). — the core-size `ratio`
  filter behind `naveja_recap`.
* Bemis, G. W. & Murcko, M. A. **The properties of known drugs. 1. Molecular
  frameworks.** *J. Med. Chem.* **39**, 2887–2893 (1996).
