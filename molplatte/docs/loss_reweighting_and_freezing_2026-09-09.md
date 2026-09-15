# Loss reweighting, and when freezing helps

> **SUPERSEDED IN PART, 2026-09-15.** The freezing conclusion below is wrong,
> and the reason is instructive: it was a LEARNING-RATE artifact.
>
> A hyperparameter sweep swept `lr` for the first time (it had been fixed at
> MolPLA's released 1.0e-3 and never examined). At the tuned 1.0e-4, freezing
> the encoder COSTS 0.0103 H@1 -- 17.8 sigma over three seeds, sd 0.0005 on
> both arms. The freeze effect reverses with lr:
>
> | lr | freeze=none | freeze=encoder | freezing |
> |---|---:|---:|---|
> | 1.0e-3 | 0.3411 | 0.3532 | **helps** +0.0121 |
> | 1.0e-4 | **0.3701** | 0.3598 | **hurts** -0.0103 |
>
> So freezing was compensating for a learning rate that was too high. The
> +0.0098 measured below is real as a measurement and wrong as a conclusion:
> it is what freezing buys you at lr 1.0e-3, not evidence that a frozen
> encoder is better. Fix the lr and the sign flips.
>
> What survives unchanged: the loss-reweighting null (§1), the test/val split
> correction (§0), and the finding that the ZINC encoder is a poor flavour
> encoder. See [hyperparameter_sweep_2026-09-15.md](hyperparameter_sweep_2026-09-15.md).

Runs of 2026-09-09, seed 911012, corpus `coconut-flavordb-full`, all scored
against the 668,913-row union library (effective vocabulary 6,916).

## Headline

1. **The linker weight is inconsequential.** It is fully controllable and moves
   nothing else. The "linker/rgroup trade-off" reported in
   [step1_zinc_transfer_2026-09-09.md](step1_zinc_transfer_2026-09-09.md) does
   not exist; the objectives are decoupled.
2. ~~**Freezing helps — if the encoder is right.**~~ **WRONG, see the banner.**
   The +0.0098 is what freezing buys at lr 1.0e-3; at the tuned 1.0e-4 it
   COSTS 0.0103 (17.8 sigma over 3 seeds). Freezing was compensating for a
   learning rate roughly 10x too high. What survives is the narrower claim it
   was built on: the ZINC encoder is a poor flavour encoder.
3. **A split mismatch invalidated the earlier A/B.** It is corrected below.

---

## 0. The measurement bug, first

Every number in `step1_zinc_transfer_2026-09-09.md` except the flavour-only
baseline came from the in-training `RGroupLibraryRetrieval/val` callback. The
flavour-only baseline `0.3098` came from a separate `run_mode=test` invocation
and is a **test**-split score. The two tables there compare test against val.

The tell was visible and was written down as a caveat rather than recognised:
the priors differ, 0.0233 against 0.0256. A prior is a property of the query
set, so two different priors mean two different query sets. That is not a
subsample nuisance, it is a different measurement.

Every checkpoint has been rescored through the *same* `run_mode=test` path.
All numbers below share one split and one prior (0.0233).

| checkpoint | H@1 | H@10 | MRR | novel@1 |
|---|---:|---:|---:|---:|
| flavour-only, 30 ep (baseline) | 0.3098 | 0.6110 | 0.4118 | 0.1134 |
| ZINC → flavour | 0.2973 | 0.5968 | 0.3977 | 0.0971 |
| ZINC → flavour, encoder frozen | 0.2328 | 0.5149 | 0.3276 | 0.0576 |
| ZINC → flavour, linker weight 0.0 | 0.3043 | 0.6026 | 0.4046 | 0.0981 |
| ZINC → flavour, linker weight 1.0 | 0.3010 | 0.6013 | 0.4021 | 0.1018 |
| flavour-enc + 20 ep, trainable | 0.3152 | 0.6087 | 0.4161 | 0.1054 |
| **flavour-enc + 20 ep, FROZEN** | **0.3250** | **0.6264** | **0.4268** | 0.1085 |

1 SE ≈ 0.0033 at N=20,000.

**This makes the ZINC verdict stronger, not weaker.** Like-for-like, ZINC
pretraining costs −0.0125 H@1 (3.8 SE), not the −0.0027 previously reported as
"within noise". The direction was right and the magnitude was understated.

---

## 1. Reweighting the linker term

Three runs, identical but for `loss_weights.linker_contrastive`, all warm-started
from `pretrain_zinc-10m_expanded24.pt`.

| weight | val linker loss | val rgroup loss | H@1 (test) |
|---:|---:|---:|---:|
| 0.0 | 4.381 | 3.4205 | 0.3043 |
| 0.1 (default) | 3.394 | 3.4234 | 0.2973 |
| 1.0 | 2.486 | 3.4341 | 0.3010 |

The weight does exactly what a weight should: linker loss spans 2.49 → 4.38, a
43% swing, monotone in the weight. And nothing else responds. `rgroup` moves
0.4%. Retrieval spans 0.007, about 2 SE, non-monotone — noise, not a trend.

**So there is no trade-off to recover.** The previous document reasoned that
because total val loss fell while the linker term rose, `rgroup` must be
"paying for" `linker` inside the sum. That inference was wrong. If the two
competed, forcing `linker` down would have pushed `rgroup` up; it does not
budge. The terms are decoupled, and the rising linker curve is cosmetic.

A weight sweep is the cheap way to tell a trade-off from a coincidence, and it
should have been run before the trade-off was asserted. Two curves moving in
opposite directions inside a sum is not evidence that either causes the other.

---

## 2. Freezing — the clean test

> **This section measures something real and draws the wrong conclusion from
> it.** Every arm here runs at lr 1.0e-3, which the 2026-09-15 sweep showed is
> roughly 10x too high. Freezing helps at that learning rate and hurts at the
> tuned one. Read the numbers; discard the recommendation.

The earlier ablation froze a ZINC-pretrained encoder and lost 6.7 points, and
that was written up as evidence against freezing. It confounds two variables:
*whether* the encoder is frozen, and *which* encoder is frozen.

Separating them needs a frozen encoder that is already correct for the domain.
Both arms below start from the same flavour-trained checkpoint
(`s1-pretrain-full-v3-s911012_best.pt`, 30 epochs) and train 20 more epochs on
the same corpus, differing only in `freeze`:

| | H@1 | vs baseline | vs each other |
|---|---:|---:|---:|
| baseline (the warm start itself) | 0.3098 | — | |
| + 20 ep, encoder trainable | 0.3152 | +0.0054 (1.7 SE) | |
| + 20 ep, encoder **frozen** | 0.3250 | +0.0152 (4.6 SE) | **+0.0098 (3.0 SE)** |

The unfrozen arm is the control for "20 more epochs". It gains little, so the
frozen arm's advantage is not simply extra training: **+0.0098 is attributable
to freezing**, with everything else held fixed.

Both arms early-stopped at ~11 epochs, so this is not a case of one arm
training longer.

### What the earlier ablation actually showed

Restated with the confound separated:

| frozen encoder | H@1 | |
|---|---:|---|
| ZINC-trained | 0.2328 | −0.077 vs baseline |
| flavour-trained | 0.3250 | +0.015 vs baseline |

Same freezing mechanism, opposite sign. Freezing is not harmful; freezing the
*wrong representation* is harmful. And the previous document's own
representation-ceiling analysis explains why it should be: a frozen encoder
bounds the hypothesis space the heads optimise within, which is a penalty when
the fixed embeddings are poor and a regulariser when they are good.

The 62%-as-much-improvement and epoch-12 saturation measured for the ZINC-frozen
run are still correct as descriptions of *that* run. They were over-generalised
into a claim about freezing.

### Caveats

Single seed. +0.0098 at 3.0 SE is suggestive, not settled — the SE describes
query sampling, not seed-to-seed variance, which is typically larger. A 3-seed
repeat is the check worth running before this drives a design decision.

> **Both halves of that caveat turned out wrong, 2026-09-15.** Seed variance
> was measured at sd 0.0005-0.0021 -- SMALLER than the 0.0033 query-sampling
> SE, not larger. And the repeat that mattered was not more seeds at this
> learning rate but a different learning rate: the confound was lr, and no
> amount of seed replication at 1.0e-3 would have exposed it.

The frozen arm is also 40% faster per epoch, so if it holds up it is cheaper as
well as better.

---

## 3. Assembly is unaffected by any of this

| run | exact | control | head gap |
|---|---:|---:|---:|
| flavour-enc frozen | 0.969 | 0.993 | 0.024 |
| flavour-enc trainable | 0.972 | 0.993 | 0.021 |
| linker 0.0 | 0.962 | 0.993 | 0.031 |
| linker 1.0 | 0.965 | 0.993 | 0.028 |

All within 1 point, control pinned at 0.993 throughout. Assembly does not care
about the retrieval loss weights or about freezing, consistent with it being
chemistry-general rather than domain-specific.

---

## 4. What changes downstream

- ~~**Step 2 should test `freeze=[encoder]` from a flavour-trained warm start.**~~
  **Superseded 2026-09-15: do NOT freeze.** At the tuned lr it costs 0.0103
  (17.8 sigma). `step1_zinc_transfer` §5's original "never freeze" advice lands
  in the right place, though for a different reason than it gave.
- **Drop the linker-reweighting thread.** It was flagged as "the most actionable
  open signal"; it was measured and it is inert.
- **Report one split.** The mixed-split comparison survived review because the
  priors were quoted as a caveat instead of being read as a split identifier.
  Any future comparison should assert matching priors before comparing.

## Reproducing

```bash
# the three reweighting / freezing arms
EXP=lw-linker1.0 ./run_experiment.sh step2      # after setting the weight in config
FREEZE=encoder EXP=freeze-flavourenc ./run_experiment.sh step2

# matched-split scoring, the part that matters
cd molplatte/src && WANDB_MODE=disabled python3 run.py run_mode=test \
  +checkpoint_path=<ckpt> assembly.enabled=true \
  data_module_kwargs.condvec_dim=24 nnet_module_kwargs.condvec_dim=24 \
  rgroup_library.vocab_path=<union vocab>
```

wandb: project `molplatte`, group `loss-reweight`.
