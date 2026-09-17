# Typed non-covalent interactions do not beat the sequence pocket embedding

**Do not build an NCI-based pocket encoder for R-group retrieval.** Typed
interaction fingerprints fail the same gate the EGNN failed, and they fail it
harder: the descriptor carries **no cross-family signal at all** (-0.0014,
-0.3 SE) even when computed from the crystal ligand -- that is, even with the
answer in hand.

## What was measured

ProLIF 2.2.1 typed interactions over all 1,255 tastepocket records, scored on
the identical probe used for ESM-2 and the geometry descriptors: for each
held-out pocket take the nearest OTHER pocket by cosine, restricted to a
different receptor AND a different ligand, then compare its ligand to the true
one by Morgan/Tanimoto. Delta is against a random pocket from the same pool.

Four representations, split by whether they could ever be deployed:

| | what it is | deployable |
|---|---|---|
| `apo-pharm` | pharmacophore class x distance shell; what the pocket *presents* | yes |
| `apo-shell` | residue type x distance shell (reimplementation) | yes |
| `holo-ifp` | ProLIF typed interactions, residue type x interaction type | **no -- oracle** |
| `holo-itype` | the same, collapsed to interaction-type totals | **no -- oracle** |

The holo rows are computed **from the crystal ligand**, which is the thing the
probe is trying to predict. They are circular by construction -- the same leak
as the condition vector in `1a6c709` -- and are reported only as a ceiling.

## Results

| representation | nearest | random | delta | SE | cross-family | SE |
|---|---:|---:|---:|---:|---:|---:|
| **ESM-2 1280 (sequence)** | 0.3590 | 0.1225 | **+0.2365** | 24.1 | **+0.0489** | 7.9 |
| shell — published control | 0.3327 | 0.1225 | +0.2102 | 22.3 | +0.0346 | 6.0 |
| comp — published control | 0.3418 | 0.1225 | +0.2193 | 22.6 | +0.0195 | 4.1 |
| shape — published control | 0.3019 | 0.1225 | +0.1795 | 18.8 | +0.0456 | 7.6 |
| apo-shell | 0.2824 | 0.1225 | +0.1600 | 17.9 | +0.0226 | 4.2 |
| apo-pharm | 0.2643 | 0.1225 | +0.1419 | 15.8 | +0.0256 | 4.7 |
| apo shell+pharm | 0.2869 | 0.1225 | +0.1644 | 18.2 | +0.0307 | 5.5 |
| holo-itype *(oracle)* | 0.1609 | 0.1225 | +0.0384 | 5.3 | +0.0100 | 2.5 |
| **holo-ifp *(oracle)*** | 0.2696 | 0.1225 | +0.1471 | 15.8 | **-0.0014** | -0.3 |
| holo-ifp + ESM-2 *(oracle)* | 0.3580 | 0.1225 | +0.2355 | 24.1 | +0.0530 | 8.3 |
| shell + apo-pharm | 0.3101 | 0.1225 | +0.1876 | 20.4 | +0.0365 | 6.2 |
| ESM-2 + apo-pharm | 0.3599 | 0.1225 | +0.2374 | 24.4 | +0.0494 | 7.9 |
| ESM-2 + holo-itype *(oracle)* | 0.3614 | 0.1225 | +0.2390 | 24.2 | +0.0467 | 7.5 |

**Nothing clears ESM-2, including the oracles.** The best oracle reaches 62% of
ESM-2's unseen-ligand delta and *none* of its cross-family delta.

### The harness was validated before the table was read

The September descriptors were re-scored through this probe code and reproduce
their published numbers within 0.007 -- the residual is the one pocket lost to a
sanitisation failure (1,254 of 1,255):

| | published | here | diff |
|---|---:|---:|---:|
| ESM-2 | +0.2435 / +0.0476 | +0.2365 / +0.0489 | 0.007 / 0.001 |
| shell | +0.2177 / +0.0325 | +0.2102 / +0.0346 | 0.008 / 0.002 |
| comp | +0.2256 / +0.0168 | +0.2193 / +0.0195 | 0.006 / 0.003 |
| shape | +0.1868 / +0.0437 | +0.1795 / +0.0456 | 0.007 / 0.002 |

My `apo-shell` reimplementation scores +0.1600 against the original's +0.2102,
so it is a **worse descriptor**, not a broken harness -- it bins to the ligand
centroid over 4 shells where the original used 3 bins and (apparently) distance
to the nearest ligand atom. The published arrays are used as the geometry
baseline throughout; `apo-shell` is retained only to document the gap.

## Caveat on the "does it add anything" rows

Concatenation is scored after per-column z-scoring, so appending 24 or 300 dims
to a 1280-d vector barely moves the cosine. `ESM-2 + holo-ifp` being flat is
therefore **weak** evidence -- dilution, not necessarily absence of signal.
`shell + apo-pharm` scoring *below* `shell` alone (+0.1876 vs +0.2102) is the
same artifact and should not be read as the pharmacophore hurting.

The load-bearing evidence is the **standalone** `holo-ifp` row: -0.0014 at
-0.3 SE cross-family. That is not a dilution effect. An oracle descriptor built
from the true ligand transfers nothing across a receptor family boundary.

## Unrelated finding: the cross-family signal is fragile

283 of 1,254 pockets use PDB-perceived bond orders because the deposited ligand
is incomplete (220 are one lipid, `6OU`, whose tails are unmodelled). Dropping
them:

| | cross-family, all 1,254 | cross-family, clean 971 |
|---|---:|---:|
| ESM-2 | +0.0489 (7.9 SE) | **+0.0023 (0.9 SE)** |
| shell | +0.0346 (6.0 SE) | +0.0105 (4.1 SE) |
| apo shell+pharm | +0.0307 (5.5 SE) | +0.0146 (4.6 SE) |
| holo-ifp *(oracle)* | -0.0014 (-0.3 SE) | +0.0021 (0.8 SE) |

**ESM-2's cross-family advantage does not survive the subset**, dropping to
insignificance, while the geometry and pharmacophore descriptors stay
significant and overtake it. This is a subset with different pool composition
(the random baseline moves 0.1225 -> 0.0970), so it is not a clean ablation and
does not retract the September result. But it does say the cross-family claim
rests on which pockets are in the set, and any future pocket work should
re-measure it on a curated set rather than inherit +0.0476 as settled.

## Why this was nearly a fabricated result

Three defects, each of which completed with rc=0 and wrote a plausible file:

1. **ProLIF spawns a process pool per complex.** 10x the cost of the work.
2. **`Chem.AddHs` gives new hydrogens no `PDBResidueInfo`.** ProLIF splits the
   protein by residue and drops atoms it cannot place, deleting every explicit
   H. The donor SMARTS `[$([O,S,#7;+0])...]-[H]` then matches nothing, so
   HBDonor/HBAcceptor were **identically zero across all 120 test pockets**
   while the run reported 120/120 success.
3. **RDKit's PDB reader leaves Asp/Glu/Arg/Lys neutral** and corpus SMILES are
   neutral forms, so Anionic/Cationic could never fire at any geometry.

What caught them was not the failure count -- that was a clean 120/120 -- but
the **interaction histogram**. A protein pocket that makes zero hydrogen bonds
is impossible. Without that check this file would have reported a
hydrophobic-contact descriptor as an NCI probe. ProLIF's own demo data was the
control separating "my prep is wrong" from "ProLIF is misconfigured".

A fourth defect cost coverage rather than correctness: `removeHs=True` is
**silently ignored when `sanitize=False`**, so crystallographic H were counted
against heavy-atom-only SMILES templates and 398 records (32%) failed bond-order
assignment. Stripping H at selection time recovered 857 -> 1,254.

## Reproduce

```bash
conda activate molplatte-ifp                       # prolif 2.2.1, MDAnalysis 2.10
python molplatte/src/scripts/build_ifp_pockets.py  # -> /tmp/ifp_pockets.npz
python molplatte/src/scripts/ifp_pocket_probe.py
```
