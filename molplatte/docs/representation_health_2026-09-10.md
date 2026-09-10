# Representation health across the training stages

Measured by the `RepresentationHealth` callback, which implements Table 1 of
Lee et al., *Understanding and Tackling Over-Dilution in Graph Neural Networks*
(KDD 2025). All scores are normalised so **higher is worse**: 0 healthy, 1 fully
degenerate.

## Headline

1. **Nothing is pathological for retrieval.** MAD stays well clear of collapse
   and feature correlation is low in every run.
2. **The ZINC encoder over-smooths noticeably more** than the flavour one —
   41% MAD decline against 33% — which is consistent with, though not proof of,
   its worse frozen retrieval.
3. **The learned fusion MLP is sitting on the uniform baseline it was supposed
   to beat.** `dilution_intra` is 0.98–0.99 in every run and drifts *upward*
   during training. This is the one number that suggests a design change, and
   it is independent of corpus, of ZINC, and of pockets.

---

## 1. Where each stage lands

Training-time figures, each on its own corpus, at the last epoch where the
Jacobian tiers ran.

| run | corpus | MAD in → out | decline | collapse | over-smoothing | over-corr |
|---|---|---|---:|---:|---:|---:|
| STEP 1 ZINC pretrain | zinc-10m | 0.873 → 0.518 | **41%** | 0.528 | **0.407** | 0.152 |
| STEP 1 flavour-only (v3) | coconut-flavordb-full | 0.734 → 0.491 | 33% | 0.507 | 0.331 | 0.138 |
| STEP 2 freeze-flavourenc | coconut-flavordb-full | 0.731 → 0.493 | 33% | 0.488 | 0.325 | 0.137 |

| run | dilution_intra | dilution_inter | over-squashing |
|---|---:|---:|---:|
| STEP 1 ZINC pretrain | 0.980 | 0.838 | 0.787 |
| STEP 1 flavour-only (v3) | 0.987 | 0.814 | 0.792 |
| STEP 2 freeze-flavourenc | 0.987 | 0.818 | 0.787 |

Corpus-matched comparison of the two shipped checkpoints, both scored on
`tastepocket_corpus` (structural tier only — see §4):

| checkpoint | MAD in → out | collapse | over-smoothing | over-corr |
|---|---|---:|---:|---:|
| freeze-flavourenc | 0.658 → 0.503 | 0.268 | 0.236 | 0.165 |
| s3-pocket-supervised | 0.658 → 0.503 | 0.259 | 0.236 | 0.165 |

The identical MAD curves are **correct, not a copy-paste**: the
pocket-supervised run froze the encoder, and these metrics are measured on the
encoder. Matching numbers are a check passing.

## 2. ZINC over-smooths more

The ZINC encoder starts far more spread out (MAD 0.873 against 0.734) — ZINC is
chemically much more diverse than flavour compounds, so that is expected — and
then gives it up faster: 41% decline against 33%, over-smoothing 0.407 against
0.331.

That sits alongside the retrieval result in
[step1_zinc_transfer](step1_zinc_transfer_2026-09-09.md): frozen, the ZINC
encoder scores **0.2328** on tastepocket against the flavour encoder's
**0.3250**. A representation that discriminates less at the output layer is a
plausible mechanism for that gap.

**Stated as correlation, not cause.** One seed, two runs, and over-smoothing was
never manipulated independently — nothing here rules out the domain-shift
explanation the transfer document already gives, and the two are not exclusive.

## 3. Training fixes over-smoothing. It does not fix dilution.

Over the 30 epochs of flavour pretraining, over-smoothing falls monotonically
and substantially:

    0.565  0.551  0.525  0.503  0.491  0.473  0.460  0.444  0.436  0.423
    0.414  0.403  0.398  0.395  0.385  0.375  0.376  0.366  0.368  0.359
    0.351  0.352  0.345  0.342  0.339  0.337  0.335  0.327  0.333  0.331

Intra-node dilution does the opposite. Sampled every fifth epoch:

    0.978  0.984  0.987  0.988  0.989  0.987

**1.0 is the pathological limit.** It means the attribute-influence
distribution is perfectly uniform — the `1/|T_v|` regime that Lee et al. assume
when `h⁰ = Σ z_t`.

This callback exists precisely because MolPLAtte *should* be able to beat that.
It inherits MolDAM's VanillaGNN, which **concatenates** its node-attribute
embeddings and learns a fusion MLP, so the weights **can** be non-uniform. At
0.99 they effectively are not: the fusion has collapsed onto the uniform
baseline, and further training moves it slightly *closer* rather than away.

Same value on ZINC (0.980) as on flavour (0.987), so this is a property of the
architecture, not of any corpus.

`dilution_inter` (~0.82) and over-squashing (~0.79) are likewise flat across
every run and every corpus — also architectural, and unchanged by anything
tried in this project.

### What that licenses

Supported: the learned attribute fusion is not differentiating between atom
attributes, and training does not make it start.

NOT supported: that fixing it would improve retrieval. Nothing here connects
dilution to Hit@K, and the measured retrieval bottleneck elsewhere is a
granularity mismatch, not representation quality. This is a lead, not a
diagnosis.

## 4. Methodology: test-mode numbers were fabricated

Worth recording, because the wrong numbers were plausible.

PyTorch Lightning runs `trainer.test()` inside `torch.inference_mode`, where
`torch.func.jacrev` cannot build a graph. The Jacobian tiers **did not raise** —
they returned **zeros**, which logged as `intra-H 0.000, dilution_inter 1.000`
and read exactly like a measurement of a fully degenerate representation. The
same checkpoint on the same corpus reports 0.987 / 0.221 from the validation
loop, and `oversquashing` was silently absent from the score line entirely.

Fixed 2026-09-10: grad is re-enabled where possible, and under inference mode
the tiers are SKIPPED with an explicit warning rather than emitting a number.
An absent metric is recoverable; a fabricated one is not.

A second trap, same day: the Jacobian tiers are gated to `every_n_epochs=5`, so
reading the **last** health line of a log usually finds structural metrics only.
That is a sampling gap, not a missing measurement — grep for `intra-H` rather
than reading the tail.

Two gaps remain. `s3-pocket-supervised` has no Jacobian figures at all, because
it was trained with `val_split=0` and the validation loop never ran; getting
them needs a short re-run with a validation split. And its structural figures
above come from test mode on a different corpus than the training-time rows, so
the two tables are not directly comparable to each other.

## Reproducing

```bash
cd molplatte/src && WANDB_MODE=disabled python3 run.py run_mode=test \
  +checkpoint_path=<ckpt> data_module_kwargs.dataset_version=<corpus> \
  rgroup_library.enabled=false
# structural tier only -- see §4. For the Jacobian tiers, read a TRAINING log:
grep 'intra-H' <run>.log
```
