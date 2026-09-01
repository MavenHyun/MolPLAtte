# MolPLAtte corpus format

What `preprocess_flavor.py` writes, why it writes that and not more, and how to
check a build before you ship it.

---

## Directory layout

One `.pt` file per **molecule** — not per decomposition, not per training
instance — plus two JSON sidecars at the corpus root:

```
<output-path>/
├── __meta__.json        consumer index + provenance
├── __manifest__.json    auditor record
├── 0a3/  FDB1183.pt  CNP0252853.pt  …
├── 0a4/  …
└── …                    (bucket names depend on --layout)
```

Bucketing is `molplatte_prep.preprocess.path_for(root, mol_id, layout)`:

| `--layout` | bucket | buckets | use when |
|---|---|---:|---|
| `flat` | none — `root/{mol_id}.pt` | 1 | < ~100K molecules |
| `shard2` | `mol_id[-2:]` | ≤ 100 | fixed-width ids (ZINC) |
| `shard3` | `mol_id[-3:]` | ≤ ~4K | fixed-width ids at scale |
| `hash3` | first 3 hex chars of `blake2b(mol_id, digest_size=2)` | 4096 | **FlavorDB / COCONUT — the default** |

`hash3` exists because suffix-slicing assumes fixed-width, uniformly-distributed
ids. `FDB4`, `FDB25595` and `CNP0252853` are none of those — `FDB4` is shorter
than a 3-char slice, and CID suffixes are far from uniform. A digest bucket is
uniform by construction. Whatever you choose is recorded in `__meta__.json`, and
readers must resolve paths through `path_for` rather than hardcoding a scheme.

Records are written by `writer.write_record`, which calls `lmdb_store.dehydrate`
first: every PyG `Data` becomes a plain dict of `numpy.ndarray`. Nothing in the
pickle carries a `molplatte_prep` class qualname or a torch tensor — the latter
matters because unpickling a torch tensor is not fork-safe under multi-worker
`DataLoader`s.

---

## Record schema

```python
{
    "mol_id":         "FDB1183",           # str, matches the filename stem
    "source":         "flavordb",          # "flavordb" | "coconut"
    "smiles":         "CC(=O)OC1=CC=CC=C1C(=O)O",   # post-wash canonical SMILES
    "method":         "macfrag",           # decomposition method
    "n_decomps":      7,                   # == len(decompositions)
    "original":       {...},               # portable form of the intact graph
    "decompositions": [ {...}, ... ],      # bookkeeping only — see below
    "meta":           {...},               # source provenance, verbatim
}
```

`smiles` is what the molecule looks like **after** `decompose.wash()`: largest
fragment kept (Naveja's salt heuristic), sanitized, neutralised, and — unless
`--no-keep-stereo` was passed — stereochemistry preserved. Atom indices
everywhere else in the record refer to *this* molecule, so re-parsing `smiles`
does not necessarily reproduce the indexing; use `original` for that.

`meta` is `SourceRecord.meta` passed straight through: `{"cid", "flavor_profile",
"InChIKey", "MolecularFormula", "MolecularWeight"}` for FlavorDB (only the keys
that were populated), `{"identifier"}` for COCONUT (the un-stripped
`CNP0252853.1` form). Nothing in the pipeline reads it; it is there so a corpus
can be traced back to its row without re-opening the source.

### `original` — the intact graph

`mol_features.data_to_portable(mol_to_pyg(mol))`, i.e. a dict with:

| key | dtype | shape | meaning |
|---|---|---|---|
| `__t` | str | — | `"MolPLAtteData_v2"` format marker |
| `__num_nodes`, `num_nodes` | int | — | atom count |
| `edge_index` | int64 | `[2, 2·nBonds]` | bidirectional edge list |
| `atomic_num`, `formal_charge`, `chiral_tag`, `hybridization`, `num_explicit_hs`, `is_aromatic` | uint8 | `[n]` | the six node feature channels |
| `bond_type`, `edge_is_aromatic`, `is_conjugated`, `bond_dir`, `bond_stereo` | uint8 | `[2·nBonds]` | the five edge feature channels |
| `is_linker`, `edge_is_linker` | bool | `[n]`, `[2·nBonds]` | all `False` on `original` |
| `linker_id`, `linker_atom`, `linker_metas` | int64 / dict | — | joint bookkeeping, empty on `original` |

Values are **indices into the tables in `mol_features.RDKIT_FEATURES`**, not raw
chemistry, and `MASK_VALUES[k] == len(RDKIT_FEATURES[k])` is reserved as the mask
index for a masked linker joint. `uint8` is deliberate: `atomic_num`'s mask index
is 128, which overflows `int8` and wraps to −128.

> **This is why `rdkit` carries an upper version bound.** Five of the eleven
> tables (`chiral_tag`, `hybridization`, `bond_type`, `bond_dir`, `bond_stereo`)
> are built by enumerating an RDKit enum at import time. If a release adds or
> reorders an enum member, every stored index shifts and a model trained on the
> old corpus reads the wrong chemistry — no exception, no warning. Re-run the
> round-trip check below after any RDKit bump.

### `decompositions[i]` — bookkeeping, not graphs

```python
{
    "core_smiles": "",                     # "" for the fragment family (see below)
    "core_atoms":  (0, 1, 2, 5, 6, 7),     # atom indices of the core, sorted
    "n_rgroups":   3,                       # == len(rgroups)
    "rgroups": [
        {"rgroup_atoms": (8, 9, 10),        # atom indices of this R-group
         "core_linker":   7,                # core-side atom of the cut bond
         "rgroup_linker": 8},               # R-group-side atom of the cut bond
        ...
    ],
    "rgroup_hashes":   ["1f0c…", "9a41…", "1f0c…"],   # list[str], parallel to rgroups
    "rgroup_condvecs": array(shape=(3, 97), dtype=uint8),  # row i ↔ rgroups[i]
}
```

The first four keys come from `molpla_instance.decomposition_record`; the last
two are added by the driver. `rgroup_hashes` and `rgroup_condvecs` are *parallel
to* `rgroups`, not nested inside it — they are dense arrays the training side
indexes by position, and keeping them flat avoids 3 dict lookups per R-group per
`__getitem__`.

Invariants a valid record satisfies:

* `core_atoms` and every `rgroup_atoms` are disjoint, and their union is
  `range(num_nodes)`.
* `core_linker ∈ core_atoms`; `rgroup_linker ∈ rgroup_atoms`; the pair is a real
  bond of `original`.
* `core_linker` values are **distinct across the R-groups of one decomposition**
  — geminal joints (two R-groups on one core atom) are rejected at decomposition
  time, because `graph_ops.detach_rgroups_multi` represents at most one joint per
  template atom and MolPLA's `is_linker` is one bit per atom.
* `1 ≤ n_rgroups ≤ --max-rgroups`, and `len(rgroup_hashes) ==
  rgroup_condvecs.shape[0] == n_rgroups`.

**Why the hash and condvec can be precomputed at all.** A detached R-group graph
depends only on its own atoms plus the masked clone of its core-side neighbour —
*not* on which sibling R-groups happen to be detached in a given `islinked` draw.
Both are therefore `islinked`-invariant, and paying for them once here saves
paying for them on every `__getitem__`.

`rgroup_hashes[i]` is `graph_hash.subgraph_hash` of R-group *i* after a
single-R-group `detach_rgroups_multi`: a 32-hex-character Weisfeiler-Lehman
graph-isomorphism hash over the node/edge feature tuples plus the linker flags.
It is the retrieval vocabulary key *and* the multi-positive grouping key for the
R-group contrastive loss — two structurally identical R-groups from different
molecules must collide, and they do, because `linker_id` is excluded from the
hash. A failed detach stores `""` rather than aborting the molecule; treat an
empty hash as "not retrievable" downstream.

`rgroup_condvecs[i]` is the condition vector of R-group *i*. It is computed from
a **real sub-molecule** — `Chem.MolFragmentToSmiles(mol, atomsToUse=rgroup_atoms)`
re-parsed — not from the masked PyG graph, because the `fr_*` counters need real
chemistry and a masked linker atom is not a real atom. Stored as `uint8`: the
vector is binary presence bits, so `float32` would cost 4× the bytes for the same
information. Under `--condvec-mode neutral` the width is 97 (85 RDKit
`fr_*` counters + 12 flavor SMARTS). Under `pocket` it is **currently all zeros**,
which is the paper's `Cond. None` ablation — see the README; that is a deliberate
loud failure, not a placeholder to be ignored.

`core_smiles` is left `""` for the fragment-family methods (`macfrag`, `synton`):
computing it costs a SMILES round-trip per candidate core, and at ~64 candidate
cores per molecule that dominates the per-molecule cost for something nothing
downstream reads. Anchored methods fill it in because they get it for free.

---

## Why bookkeeping and not detached graphs

MolPLA's training instance is a `(decomposition, islinked)` pair: `islinked` is a
bitmask over the decomposition's R-groups where `1` means "stays attached to the
core" and `0` means "detached, and therefore a retrieval target". At least one
R-group must be detached, so a k-R-group core admits **`2^k − 1`** instances.

At the measured mean of **3.34 R-groups per core** (MacFrag re-framed at
`--core-ratio 0.5`), that is `2^3.34 − 1 ≈ 9.1` instances per decomposition.
Materialising each one as `(P, R…)` graphs would multiply the corpus by **~9×**
while adding zero information: every instance is a deterministic function of
`original` plus the three index fields already stored.

So the corpus stores the intact graph once and the indices once. The training-time
dataset does:

```python
def __getitem__(self, idx):
    rec  = load(self.ids[idx])                      # one .pt, O(1) I/O
    d    = random.randrange(rec["n_decomps"])       # one decomposition
    dec  = rec["decompositions"][d]
    mask = sample_islinked(dec["n_rgroups"])        # one of the 2^k − 1
    return build_instance(hydrate(rec["original"]), decomposition_from(dec), mask,
                          mol_id=rec["mol_id"], decomp_idx=d)
```

`build_instance` (`molplatte_prep/molpla_instance.py`) calls
`detach_rgroups_multi` on the spot and returns `MolPlaInstance(G, P, R, …)` —
`G` the intact graph with linker flags cleared, `P` the core plus still-attached
R-groups carrying one masked linker atom per detachment, `R` one graph per
detached R-group, plus `instance_id = "{mol_id}#{decomp_idx}-{bitstring}"`
mirroring MolPLA's `data_instance_id`. `__len__` is the number of *molecules*, so
an epoch sees each molecule once through a different random slice of its
`2^k − 1` space and the full space is covered across epochs. This is MolDAM's
online-sampling idiom applied to MolPLA's subset space.

Cost of doing it online: one `detach_rgroups_multi` per `__getitem__`, which is
graph slicing on a ≤50-atom molecule — negligible against the GNN forward pass,
and it happens in a DataLoader worker. Pass `compute_hashes=False` to
`build_instance` when the precomputed `rgroup_hashes` are enough, which they
usually are.

---

## `__meta__.json` — the consumer index

Written by `writer.write_meta`, atomically. Everything a `Dataset` needs to build
its sampler index **without opening a single record**, followed by the full
provenance block:

```json
{
  "n_records": 25412,
  "ids":       ["FDB1183", "FDB1234", "..."],
  "n_decomps": [7, 3, "..."],
  "n_rgroups_per_decomp": [[3, 2, 4, 2, 3, 5, 2], [2, 2, 3], "..."],

  "molplatte_prep_version": "0.1.0",
  "source": "flavordb",
  "method": "macfrag",
  "method_family": "fragment",
  "decomposition_params": {},
  "core_ratio": 0.5,
  "max_cores": 10,
  "max_rgroups": 8,
  "keep_stereo": true,
  "size_filter": {"min_heavy_atoms": 5, "max_heavy_atoms": 50},
  "condvec_mode": "neutral",
  "condvec_dim": 97,
  "sample_fraction": null,
  "sample_seed": 42,
  "layout": "hash3",
  "params_provenance": "written at preprocessing time by preprocess_flavor.py",
  "created_at": "2026-08-18T06:11:04"
}
```

`ids`, `n_decomps` and `n_rgroups_per_decomp` are **parallel arrays of equal
length** — `write_meta` raises if they disagree, and the check below re-asserts
it. `ids` is sorted, and `n_rgroups_per_decomp[i][j]` lets a sampler weight
molecules by their true instance count (`Σ_j 2^k_ij − 1`) without touching disk.

`decomposition_params` holds the method-specific kwargs actually passed to the
decomposer: `{"ratio": …, "include_ring": …}` for `naveja_recap`, `{}` for the
others (which take their partition defaults, with `core_ratio` applied later by
`partitions_to_decompositions`). Note that `--core-ratio` therefore means two
different things depending on family — the Naveja core-size threshold for
`naveja_recap`, the core-subset threshold in the fragment-tree re-framing for
`macfrag`/`synton`.

The provenance block lives *inside* the corpus deliberately: MolDAM_prep lost the
run manifest for seven corpora during a directory move and had to reconstruct
decomposition ratios from file mtimes. A corpus that cannot say how it was built
is a corpus you have to rebuild.

**Known gap:** the provenance block does *not* record the RDKit version the
corpus was featurised with, which is the one thing check 3 below most wants. Note
it by hand (`__manifest__.json` → `argv` plus your environment record) until the
field is added.

## `__manifest__.json` — the auditor record

Written by `writer.write_manifest`, also atomic. It repeats the whole provenance
block and adds what only that run knows. Not read by any consumer; it exists so a
human can answer "what actually happened during that run":

```json
{
  "…": "every provenance key above, repeated",
  "argv": ["--source", "coconut", "--method", "macfrag", "..."],
  "workers": 64,
  "batch_size": 200,
  "total_mols_processed": 467812,
  "n_records": 409578,
  "n_resumed": 0,
  "status_counts": {"ok": 409578, "no_decomp": 57030, "wash_failed": 1204},
  "elapsed_seconds": 12043.7,
  "throughput_mol_per_s": 38.84,
  "completed_at": "2026-08-18T09:31:47"
}
```

`status_counts` is the block to read first. Every molecule lands in exactly one
bucket, and the failure buckets carry the exception class so a systematic problem
is visible without re-running:

| status | meaning |
|---|---|
| `ok` | record written |
| `wash_failed` | `wash()` returned `None` — unparseable or unsanitizable |
| `mol_to_pyg_failed:<Exc>` | featurisation raised |
| `decompose_failed:<Exc>` | the decomposer raised |
| `no_decomp` | decomposer ran and returned nothing |
| `write_failed:<Exc>` | disk/serialisation error |
| `unhandled:<Exc>` | caught by the worker's catch-all — should be zero |

`no_decomp` is the count the README's "no-decomp" column predicts: ~12% for
`macfrag`, ~18% for `bemis_murcko`, ~50% for `synton`. If it comes back at 50%
you ran `synton`; if it comes back near zero on `macfrag`, check that the size
filter is not silently rejecting everything upstream instead.

Rule of thumb: `__meta__.json` is machine-read and must stay parseable per
training run; `__manifest__.json` is human-read and may grow.

---

## Verification recipe

Run all three levels after a build. Level 1 is seconds, level 2 is a minute,
level 3 is only needed after an RDKit bump.

### 1 · Index consistency

```bash
python - <<'PY'
import json
from pathlib import Path
from molplatte_prep.preprocess import path_for

root = Path("~/preprocessed/molplatte/flavordb/macfrag").expanduser()
meta = json.loads((root / "__meta__.json").read_text())
n = meta["n_records"]

assert len(meta["ids"]) == n, "n_records disagrees with ids"
assert len(meta["n_decomps"]) == n
assert len(meta["n_rgroups_per_decomp"]) == n
assert len(set(meta["ids"])) == n, "duplicate mol_id in index"
assert all(len(r) == d for r, d in
           zip(meta["n_rgroups_per_decomp"], meta["n_decomps"]))
assert all(1 <= k <= meta["max_rgroups"]
           for row in meta["n_rgroups_per_decomp"] for k in row)
assert all(d <= meta["max_cores"] for d in meta["n_decomps"])

missing = [i for i in meta["ids"] if not path_for(root, i, meta["layout"]).is_file()]
assert not missing, f"{len(missing)} indexed records absent, e.g. {missing[:5]}"

on_disk = {p.stem for p in root.rglob("*.pt")}
orphans = on_disk - set(meta["ids"])
assert not orphans, f"{len(orphans)} unindexed .pt files, e.g. {sorted(orphans)[:5]}"

instances = sum(sum(2**k - 1 for k in row) for row in meta["n_rgroups_per_decomp"])
print(f"{n} molecules, {sum(meta['n_decomps'])} decompositions, "
      f"{instances} reachable training instances")
PY
```

An orphan `.pt` with no index entry is the signature of a run killed between
`write_record` and `write_meta`: re-run with `--resume`, which rebuilds the index
from the prior `__meta__.json` intersected with what is actually on disk.

### 2 · Record invariants on a sample

```bash
python - <<'PY'
import json, random, torch
from pathlib import Path
from molplatte_prep.preprocess import path_for

root = Path("~/preprocessed/molplatte/flavordb/macfrag").expanduser()
meta = json.loads((root / "__meta__.json").read_text())
dim  = meta["condvec_dim"]

for mol_id in random.sample(meta["ids"], min(200, meta["n_records"])):
    rec = torch.load(path_for(root, mol_id, meta["layout"]), weights_only=False)
    assert rec["mol_id"] == mol_id and rec["source"] == meta["source"]
    assert rec["method"] == meta["method"]
    assert rec["n_decomps"] == len(rec["decompositions"]) >= 1

    g = rec["original"]
    n = int(g["num_nodes"])
    assert g["atomic_num"].shape == (n,)
    assert g["edge_index"].shape[1] == g["bond_type"].shape[0]
    assert not g["is_linker"].any(), "original must carry no linker flags"

    bonds = {tuple(sorted(e)) for e in g["edge_index"].T.tolist()}
    for dec in rec["decompositions"]:
        core = set(dec["core_atoms"])
        k = dec["n_rgroups"]
        assert k == len(dec["rgroups"]) >= 1
        assert len(dec["rgroup_hashes"]) == k
        assert dec["rgroup_condvecs"].shape == (k, dim)
        seen = set(core)
        for rg, h in zip(dec["rgroups"], dec["rgroup_hashes"]):
            atoms = set(rg["rgroup_atoms"])
            assert not (atoms & seen), "R-group overlaps core or sibling"
            seen |= atoms
            assert rg["core_linker"] in core
            assert rg["rgroup_linker"] in atoms
            assert tuple(sorted((rg["core_linker"], rg["rgroup_linker"]))) in bonds
            assert h == "" or len(h) == 32
        assert seen == set(range(n)), "atoms unaccounted for"
        linkers = [rg["core_linker"] for rg in dec["rgroups"]]
        assert len(set(linkers)) == len(linkers), "geminal joint leaked through"
print("record invariants ok")
PY
```

Two things worth counting rather than asserting on: the fraction of empty
`rgroup_hashes` (a failed detach — should be ~0; anything above a handful means
a decomposer is emitting joints `detach_rgroups_multi` cannot represent), and the
fraction of all-zero condvec rows. Under `--condvec-mode neutral` an all-zero row
means an R-group with no recognised functional group, which is normal for bare
alkyl; under `pocket` **every** row is zero by construction, and a non-zero one
means a pocket encoder was wired in without the metadata being updated.

### 3 · Round-trip after an RDKit bump

The feature indices are only meaningful against the RDKit that built them. Before
trusting a corpus under a new RDKit, confirm the enum tables still line up:

```bash
python - <<'PY'
import json, random, torch
from pathlib import Path
import rdkit
from rdkit import Chem
from molplatte_prep.mol_features import RDKIT_FEATURES, mol_to_pyg, data_to_portable
from molplatte_prep.preprocess import path_for

root = Path("~/preprocessed/molplatte/flavordb/macfrag").expanduser()
meta = json.loads((root / "__meta__.json").read_text())
print(f"running rdkit {rdkit.__version__}; table sizes now:")
print({k: len(v) for k, v in RDKIT_FEATURES.items()})
# Expected at rdkit 2026.03.2: chiral_tag 9, hybridization 9, bond_type 22,
# bond_dir 7, bond_stereo 8.  Any change here invalidates the corpus.

bad = 0
sample = random.sample(meta["ids"], min(500, meta["n_records"]))
for mol_id in sample:
    rec = torch.load(path_for(root, mol_id, meta["layout"]), weights_only=False)
    mol = Chem.MolFromSmiles(rec["smiles"])
    if mol is None:
        bad += 1
        continue
    fresh = data_to_portable(mol_to_pyg(mol))
    for key in ("atomic_num", "chiral_tag", "hybridization",
                "bond_type", "bond_dir", "bond_stereo"):
        if fresh[key].shape != rec["original"][key].shape or \
           not (fresh[key] == rec["original"][key]).all():
            bad += 1
            break
print(f"{bad} / {len(sample)} molecules re-featurise differently")
PY
```

`0 / 500` means the tables are stable and the corpus is safe to reuse. Anything
above zero means the corpus's embedding indices no longer mean what the new RDKit
thinks they mean — rebuild rather than retrain, and raise the pin in
`pyproject.toml` only after the rebuild.

(The re-featurisation goes through `smiles`, so it also depends on canonical
SMILES output being stable. A non-zero count is therefore a signal to
investigate, not proof of an enum shift; check the printed table sizes first —
those move only when an enum actually changes.)

---

## Reading a corpus

```python
import json, torch
from pathlib import Path
from molplatte_prep import build_instance, sample_islinked
from molplatte_prep.mol_features import portable_to_data
from molplatte_prep.preprocess import path_for

root = Path("~/preprocessed/molplatte/flavordb/macfrag").expanduser()
meta = json.loads((root / "__meta__.json").read_text())

rec = torch.load(path_for(root, meta["ids"][0], meta["layout"]), weights_only=False)
G   = portable_to_data(rec["original"])
dec = rec["decompositions"][0]
inst = build_instance(G, decomposition_from(dec), sample_islinked(dec["n_rgroups"]))
```

`decomposition_from` is the training repo's adapter that turns the stored dict
back into a `decompose.Decomposition` — `core_smiles`, `core_atoms`, and one
`RGroupInfo(rgroup_atoms, core_linker, rgroup_linker)` per entry of `rgroups`. It
is deliberately *not* in this package: preprocessing's contract stops at the
on-disk dict, and the training repo owns how it hydrates.
