# Does 3D geometry beat the sequence pocket embedding?

Run 2026-09-09 on the 1,255 tastepocket sites, scored with the probe from
[step3_pocket_capacity](step3_pocket_capacity_2026-09-09.md): for each held-out
pocket, take the nearest OTHER pocket by cosine — restricted to a different
receptor AND a different ligand — and compare its ligand to the true one by
Morgan/Tanimoto. No training anywhere.

## Headline

**No 3D representation clears the sequence baseline, so a trained EGNN is not
justified for R-group retrieval. But geometry transfers across receptor families
where residue composition does not, which is the opposite of what the headline
number suggests.**

| representation | nearest | random | delta | SE | cross-family | SE |
|---|---:|---:|---:|---:|---:|---:|
| **ESM-2 1280 (sequence)** | 0.3588 | 0.1153 | **+0.2435** | 24.8 | **+0.0476** | 7.3 |
| comp — residues, NO geometry | 0.3410 | 0.1153 | +0.2256 | 23.1 | +0.0168 | 3.7 |
| shell — residues × distance | 0.3330 | 0.1153 | +0.2177 | 22.9 | +0.0325 | 5.6 |
| shape — geometry, no identity | 0.3021 | 0.1153 | +0.1868 | 19.7 | +0.0437 | 7.9 |
| EGNN — random weights | 0.2971 | 0.1153 | +0.1817 | 19.6 | +0.0427 | 7.4 |

The four 3D representations exist so that a null is interpretable. An untrained
EGNN alone cannot separate "geometry is uninformative" from "these weights
learned nothing", so `comp` (composition with no 3D at all) and `shape`
(geometry with no residue identity) bracket it.

## 1. Most of the sequence signal is composition, not geometry

`comp` is a 20-vector of residue-type counts. It has no coordinates, no
ordering, no structure — and it recovers **93%** of ESM-2's headline delta
(+0.2256 against +0.2435). Nearly all of what the probe measured in
`step3_pocket_capacity` is available from *which residues line the pocket*.

That reframes the earlier result. The finding "the ESM-2 pocket space predicts
ligand chemotype across unseen receptors" is true, but the mechanism is largely
amino-acid composition rather than anything a 650M-parameter language model
learned about folding.

## 2. Across families, geometry is what survives

The cross-family column reverses the ordering:

| | cross-family delta | vs comp |
|---|---:|---:|
| comp (composition) | +0.0168 | — |
| shell (comp + distance) | +0.0325 | 1.9× |
| shape (pure geometry) | +0.0437 | **2.6×** |
| EGNN (random weights) | +0.0427 | 2.5× |
| ESM-2 | +0.0476 | 2.8× |

Composition is family-specific: related receptors are lined by related residues,
so `comp` scores well within a family and collapses across one. Geometry does
not collapse. A pocket's shape carries something about the chemistry it binds
that survives a family boundary, and **a randomly initialised EGNN gets within
10% of a pretrained 650M protein language model on that split**.

Adding distance bins to composition (`shell`) recovers about half the gap,
which is the same effect seen a second way.

### Not a size artifact

Bigger pockets hold bigger ligands, and two large molecules share more Morgan
bits, so a size-only descriptor could manufacture this. It does not:

| | cross-family delta |
|---|---:|
| pocket size alone (one number) | **−0.0088** (−2.4 SE) |
| shape, full | +0.0457 |
| shape, size/count columns removed | +0.0244 (4.9 SE) |

Size alone is slightly *negative*. Geometry stripped of size still beats
composition. The effect is shape, not scale.

## 3. What this licenses

**Do not train an EGNN for R-group retrieval.** The criterion set before running
this was that a geometric representation should clear the sequence baseline
before earning a training run, and none does. The stronger argument is upstream:
ESM-2's *larger* +0.2435 translated into only +0.016 H@1 (1.8 SE) on actual
R-group retrieval under nested selection. The binding constraint is the
granularity mismatch — chemotype-level signal against an exact-WL-hash target —
and a better pocket encoder does not touch it.

**But record the cross-family result**, because it is the one thing here that
points somewhere. If the retrieval target is ever coarsened to chemotype level,
geometry — not sequence, and not composition — is the representation that
generalises to unseen receptor families, and it does so untrained.

Two caveats. These are untrained descriptors; a trained EGNN could be better or
worse and this cannot say which. And 42 CV components over 17 families is a
small basis for a claim about family transfer — the SEs are computed over 1,255
sites, but those sites are not independent.

## Reproducing

```bash
python3 build_geom_pockets.py /tmp/geom_pockets.npz   # ProDy, matches pockets.jsonl
python3 egnn_probe.py                                  # all five representations
```

The extraction must use the project's ProDy parser, not gemmi: chain and residue
numbering follow author numbering in `pockets.jsonl`, and gemmi's label-based
naming silently matches zero residues rather than erroring.
