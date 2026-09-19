# MolPLAtte findings index

Documents are dated by when the work was done. Several carry **retraction
banners** -- a claim that later evidence overturned is annotated in place rather
than deleted, so the reasoning chain stays auditable.

Start with **[FINDINGS.md](FINDINGS.md)** for the plain-language summary.

## Design

| doc | what it covers |
|---|---|
| [molplatte_design.md](molplatte_design.md) | architecture, losses, the logQ correction, corpus construction |

## What worked

| doc | headline |
|---|---|
| [hyperparameter_sweep_2026-09-15.md](hyperparameter_sweep_2026-09-15.md) | width 512 + lr 1e-4: H@1 0.3098 → 0.3701 (+19%), novel@10 +25%. The shipped model. |
| [step1_zinc_transfer_2026-09-09.md](step1_zinc_transfer_2026-09-09.md) | ZINC pretraining costs −0.0125 H@1; its last justification (the assembly head) is matched by flavour-only training. The stage can be retired. |
| [representation_health_2026-09-10.md](representation_health_2026-09-10.md) | collapse/dilution diagnostics; Jacobian tiers |

## What did not work: the pocket programme

Read in order -- each answers the objection raised by the previous one.

| doc | question | answer |
|---|---|---|
| [step2_pocket_results_2026-09-05.md](step2_pocket_results_2026-09-05.md) | does pocket conditioning help at width 300? | no |
| [step3_pocket_capacity_2026-09-09.md](step3_pocket_capacity_2026-09-09.md) ⚠️ | is it a capacity problem? | no — **partly superseded**, the runs were confounded |
| [egnn_pocket_probe_2026-09-09.md](egnn_pocket_probe_2026-09-09.md) ⚠️ | would 3D geometry beat the sequence embedding? | no — a random-weight EGNN gets within 90% of ESM-2 |
| [ifp_pocket_probe_2026-09-17.md](ifp_pocket_probe_2026-09-17.md) ⚠️ | would typed non-covalent interactions? | no — zero cross-family signal even as an oracle |
| [pocket_forgetting_2026-09-17.md](pocket_forgetting_2026-09-17.md) ⚠️ | was the harm catastrophic forgetting? | **yes** — freezing the flavour pathway removes it entirely, leaving pockets *inert* rather than harmful |
| [rgroup_granularity_probe_2026-09-19.md](rgroup_granularity_probe_2026-09-19.md) | is the retrieval target too coarse for a pocket? | **no** — pockets predict R-groups at 20.8σ. Refutes the explanation carried by the three ⚠️ docs above. |
| [pocket_retrieval_baseline_2026-09-19.md](pocket_retrieval_baseline_2026-09-19.md) | then why does nothing learn it? | the frequency prior already supplies most of it. The pocket's marginal value is ~+0.03 hit@5. |

⚠️ = carries a retraction banner.

## Method / controls

| doc | what it covers |
|---|---|
| [shuffle_test_2026-09-01.md](shuffle_test_2026-09-01.md) | condvec shuffle controls: separating information from capacity |
| [loss_reweighting_and_freezing_2026-09-09.md](loss_reweighting_and_freezing_2026-09-09.md) | linker loss weight is inert; freezing's apparent gain was a learning-rate artifact |

## Reproducing

```bash
# the sweep that produced the shipped checkpoint
python molplatte/src/scripts/run_sweep.py --stage W --gpu 1

# the pocket programme
python molplatte/src/scripts/egnn_pocket_probe.py
python molplatte/src/scripts/build_ifp_pockets.py && python molplatte/src/scripts/ifp_pocket_probe.py
python molplatte/src/scripts/run_pocket_forgetting_cv.py --gpu 1 --folds 0,1,2,3,4
python molplatte/src/scripts/rgroup_granularity_probe.py
python molplatte/src/scripts/pocket_retrieval_baseline.py
```

## Standing caveats on the pocket work

- single seed (911012) throughout; no seed replication on the CV ladders
- `val` and `test` are the same held-out fold, so early stopping selected on
  scored data (shared by all arms; absolute numbers optimistic)
- 17 receptor families / 96 receptors, and TRPV1 alone is 522 of 1,255 pockets
