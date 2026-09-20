# tastepocket expansion: +222 pockets, TRPV3 added, fat taste excluded

Searched the PDB for chemosensory structures the set was missing and rebuilt the
corpus. **Every build setting was matched field-for-field against the original
`__meta__.json`, not taken from script defaults** -- see the failure note below
for why that mattered.

## Result

| | before | after |
|---|---:|---:|
| pocket instances | 1,255 | **1,477** |
| PDB entries | 322 | **356** |
| receptor proteins | 96 | **106** |
| families | 17 | **19** |
| corpus records | 243 | **263** |
| decompositions | 734 | **794** |
| union R-group vocabulary | 91,935 | **91,943** |

New pockets by family: TRPV3 **144**, insect gustatory receptor 48, CaSR 16,
OTOP1/PKD2L1 (sour) 10, insect odorant receptor 2, T1R sweet/umami 2.

**TRPV3 is the substantive addition** -- a warmth/camphor chemesthesis receptor
absent from the set entirely, 12 human structures at up to 1.83 Å, and its
ligands include `A1EBV` = *(3S)-3,7-dimethyloct-6-enal* (**citronellal**) and a
second flavour aldehyde. It splits across folds 0 and 4 rather than forming
another single-family fold.

## What was deliberately excluded

**Fat taste (FFAR1/FFAR4), 36 entries.** Excluded on instruction. The structures
are real but FFAR1/4 are antidiabetic drug targets, so the deposited ligands are
GLPG-0974- and TAK-875-class compounds at MW 370-620, not dietary fatty acids.

**TAAR, 27 entries -- excluded by the EXISTING curation, not by this run.** They
were already in `taste_odor_pdb.json` at tier `T3_related_or_noise`: they are
amphetamine-, methamphetamine- and LSD-bound human TAAR1 complexes. Human TAAR1
is a neuropsychiatric target. The original curation was right and the merge left
it intact.

That is why only 36 of 81 search hits were added: **45 were already curated at a
lower tier.** The diff that produced "256 new candidates" was against extracted
pockets, not against the curation file.

## Only 8 novel R-groups from 222 new pockets

The union vocabulary grew 91,935 → 91,943. The new ligands decorate with
fragments the library already had, which independently confirms the
in-distribution finding in
[pocket_retrieval_baseline_2026-09-19.md](pocket_retrieval_baseline_2026-09-19.md):
tastepocket targets sit in the head of the frequency distribution.

## A silent misconfiguration, caught by comparing metadata

The first corpus build **completed cleanly, reported `status: ok`, and wrote 188
records with `condvec: neutral (dim 97)`** -- the legacy conditioning vector,
not the `two_part` 1304-d (24 flavour + 1280 raw ESM-2) the corpus requires. It
also deduplicated when the original did not.

Nothing errored. The corpus would have trained. Every pocket number computed
from it would have been meaningless.

It was caught by diffing the build banner against the original `__meta__.json`,
which then surfaced a second difference: `min_rgroup_atoms` defaulting to 1
where the original used 2, admitting single-atom R-groups and inflating the
vocabulary. The final build differs from the original in **0 settings**.

## Verification

- 1,255 pre-existing pockets re-embed **identically** (max |Δ| 2.1e-06, min
  cosine 0.99999988) -- the pipeline is deterministic and the expansion did not
  perturb existing data.
- v2 corpus loads: 794 items, condvec 1304, both halves populated, pocket-half
  norm 7.30 (raw ESM-2 range).
- `shuffle_pocket_only` control still valid on v2: flavour preserved 56/56,
  pocket changed 56/56.
- PCA basis orthonormal to 2.2e-15; variance retained 96.2-97.3% per fold.

## IMPORTANT: folds were reassigned

Only **108 of 269** molecules kept their previous fold -- the connected
components of the (ligand, receptor) graph change when receptors are added.
**Results computed on v2 folds are NOT comparable to the 5-fold numbers in the
pocket documents.** Any comparison must re-score the baseline on v2 folds.

## Paths (built side-by-side; originals untouched)

```
~/preprocessed/molplatte/tastepocket_v2/            pockets, ligands, labels, embeddings
~/preprocessed/molplatte/tastepocket_corpus_v2/     corpus + folds.json + rgroup_vocab
~/preprocessed/molplatte/union_vocab/base-full__crossdocked__tastepocket_v2/
~/preprocessed/molplatte/pocket_basis_v2/           PCA basis per fold
~/datasets/tastepocket/data/taste_odor_pdb.json     merged (backup: .bak-2026-09-20)
```

## Revision: insect structures removed (same day)

Insect chemoreception was dropped on instruction; the flavour work is
human-facing. Filtering is by **sensor organism, not by family label** --
see below for why that distinction was load-bearing.

| | v2 (with insect) | **v3 (final)** |
|---|---:|---:|
| pocket instances | 1,477 | **1,288** |
| PDB entries | 356 | **271** |
| receptor proteins | 106 | **73** |
| families | 19 | **15** |
| corpus records | 263 | **201** |
| decompositions | 794 | **610** |
| union R-group vocabulary | 91,943 | **91,941** |

Removed: 189 pockets across 85 PDB entries -- insect OBP 110, gustatory
receptor 48, insect OR/Orco/IR 22, CSP 9. Odour coverage takes the hit, as
expected: most odour structures were insect OBPs, leaving vertebrate lipocalins
and olfactory GPCRs.

### Two defects the filter exposed

**The family labels could not be filtered on.** 24 entries were filed under
`OBP – insect odorant/pheromone binding protein` but are **bovine, porcine,
canine, rat and human** lipocalins (1PBO, 1E06, 1OBP, 8AEH-J, 3FIQ, 4RUN, ...).
A name-based filter would have deleted genuine vertebrate odorant carriers as
"insect". They are now relabelled to the vertebrate family in
`taste_odor_pdb.json`, each carrying a `relabelled` provenance note.

**The expansion entries had no organism at all.** `extract_tastepocket_pockets`
reads `sensor_organism` -- the field naming the RECEPTOR chain as distinct from
G-protein subunits and fusion partners -- and the entries built from the PDB
search only carried `organisms`. 232 pockets, including every TRPV3 record, had
a blank organism and would have been invisible to any organism filter. Fixed by
resolving the sensor chain per entry against the family keyword while excluding
G-proteins, lysozyme, BRIL, GFP and nanobody partners.

### Verified

- corpus settings differ from the original in **0 fields**
- corpus loads: 610 items, condvec 1304
- `shuffle_pocket_only` control: flavour preserved 60/60, pocket changed 60/60
- **insect pockets remaining: 0**
- PCA basis orthonormal to 2.2e-15, variance retained 97.7-98.4%

### Final paths

```
~/preprocessed/molplatte/tastepocket_v3/          pockets, ligands, labels, embeddings, folds
~/preprocessed/molplatte/tastepocket_corpus_v3/   corpus + folds.json + rgroup_vocab
~/preprocessed/molplatte/union_vocab/base-full__crossdocked__tastepocket_v3/
~/preprocessed/molplatte/pocket_basis_v3/
```

v2 is retained as the with-insect intermediate. Originals untouched.
**Folds were recomputed again**, so v3 numbers are comparable neither to the
original 5-fold results nor to v2.

## Reproduce

```bash
python molplatte_preprocess/scripts/search_pdb_chemosensory.py
python molplatte_preprocess/scripts/build_pdb_expansion_entries.py
# then: extract_tastepocket_ligands -> extract_tastepocket_pockets ->
#       embed_pockets_esm -> build_tastepocket_dataset -> preprocess_flavor
#       -> enumerate_rgroups -> build_union_vocab -> build_pocket_basis
```
