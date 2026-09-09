# Does ZINC pretraining help? — STEP 1 → STEP 2 transfer

Runs of 2026-09-08/09, seed 911012, all scored against the same
668,913-row union library (`zincbase__flavor__crossdocked__tastepocket`,
effective vocabulary 6,916).

## Headline

**ZINC pretraining does not improve retrieval. It delivers the assembly head.**

Those are separate capabilities and the evidence separates cleanly:

| | retrieval H@1 | assembly, exact reconstruction |
|---|---:|---:|
| flavour-only (no ZINC) | **0.3098** | no assembly head exists |
| ZINC → flavour | 0.3071 | 0.966 |
| ZINC → flavour, encoder frozen | 0.2425 | **0.972** |

---

## 1. STEP 1 — pretraining on ZINC

`zinc-10m`, 9,860,230 molecules drawn flat across all 89 MW/logP tranches,
30.2M training items, condvec off (ZINC has no flavour labels and no pockets).

| K | hit@K | prior | lift |
|---|---:|---:|---:|
| 1 | 0.1109 | 0.0157 | **7.1×** |
| 10 | 0.2708 | 0.0658 | 4.1× |
| 100 | 0.4814 | 0.1677 | 2.9× |

MRR 0.1643. These are NOT comparable to the flavour runs below: ZINC's own
effective vocabulary is 6,939 against the flavour library's 889, so retrieval
there is a nearly 8× more many-way problem. Only lift over the prior travels.

### The loss curves say more than the final number

| ep | train | val | graph | linker | rgroup | assembly |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4.5314 | 4.6795 | 0.2104 | 2.8963 | 4.1701 | 0.0188 |
| 1 | 4.1410 | 4.5100 | 0.2028 | 3.0438 | 3.9947 | 0.0162 |
| 2 | 4.0115 | 4.4051 | 0.1958 | 3.1180 | 3.8900 | 0.0149 |
| *(restart, warm-started)* | | | | | | |
| 3 | 3.8379 | 4.2799 | 0.2157 | 3.1014 | 3.7470 | 0.0141 |
| 4 | 3.8389 | 4.2850 | 0.2145 | 3.3011 | 3.7337 | 0.0134 |
| 5 | 3.8154 | 4.2649 | 0.2247 | 3.1800 | 3.7157 | 0.0132 |

Two things the aggregate hides:

- **The linker-contrastive loss RISES throughout** (2.90 → 3.12, then ~3.1–3.3)
  while total val loss falls. The three objectives are summed, so `rgroup`
  improving masks `linker` degrading. This is a genuine trade-off inside the
  loss, and it is independent of ZINC — worth checking whether STEP 2 does the
  same.
- **It had flattened by epoch 3.** Train loss moved 0.006 across the last three
  epochs. More ZINC epochs would not have helped; the ceiling was reached
  early. That, rather than any forgetting story, is why the transfer is small.

---

## 2. The A/B — same library, same corpus, same seed

The flavour-only checkpoint (`s1-pretrain-full-v3-s911012`, 30 epochs, no ZINC)
rescored against the union library, beside the ZINC-pretrained run:

| metric | flavour-only | ZINC → flavour | delta | ~1 SE |
|---|---:|---:|---:|---:|
| H@1 | 0.3098 | 0.3071 | −0.0027 | 0.0033 |
| H@10 | 0.6110 | 0.6061 | −0.0049 | 0.0034 |
| H@100 | 0.8129 | 0.8119 | −0.0010 | 0.0028 |
| MRR | 0.4118 | 0.4072 | −0.0046 | — |
| base@1 | 0.3570 | 0.3554 | −0.0016 | 0.0034 |
| novel@1 | 0.1134 | 0.1048 | −0.0086 | 0.0022 |
| novel@10 | 0.3407 | 0.3155 | −0.0252 | 0.0034 |
| novel@100 | 0.6059 | 0.5810 | −0.0249 | 0.0035 |

Worse on 8 of 8. No single metric clears one standard error, so nothing here is
individually significant — but the sign is consistent, and the two largest
deltas are on `novel`, which is exactly where pretraining was supposed to help.

> An earlier reading of the STEP 2 result claimed the near-identical H@1 on a
> 7.3× larger library was "the transfer showing up". That was wrong. The
> flavour-only model handles the larger library just as well; the library was
> never the handicap it was assumed to be. The A/B is what settled it.

Caveats: single seed; 6 ZINC + 20 flavour epochs against 30 flavour epochs; the
two runs drew different 20,000-query subsamples (visible in the differing
priors, 0.0233 vs 0.0256).

---

## 3. The freeze ablation — this one is unambiguous

`freeze=[encoder]`, 5,844,464 of 6,808,535 parameters held fixed (85.8%).

| metric | flavour-only | ZINC, trainable | **ZINC, FROZEN** | frozen − baseline |
|---|---:|---:|---:|---:|
| H@1 | 0.3098 | 0.3071 | **0.2425** | **−0.0673** |
| H@10 | 0.6110 | 0.6061 | 0.5223 | −0.0887 |
| MRR | 0.4118 | 0.4072 | 0.3360 | −0.0758 |
| base@1 | 0.3570 | 0.3554 | 0.2850 | −0.0720 |
| novel@10 | 0.3407 | 0.3155 | 0.2112 | **−0.1295** |
| novel@100 | 0.6059 | 0.5810 | 0.4388 | **−0.1671** |

−0.067 H@1 is roughly **20 standard errors** — nothing like the ±0.003 wobble
above.

**This is the opposite of catastrophic forgetting.** Twenty epochs of unfrozen
flavour training are not erasing something valuable; they are REPAIRING a
representation that does not fit the target domain. Held fixed, the ZINC
encoder drags retrieval down hard.

That matches the distribution measured before the corpus was built: ZINC sits
at ~337 Da median while flavour compounds are smaller, and 62% of FlavorDB
falls below ZINC's median. The damage concentrates on `novel` chemistry
(−0.130, −0.167, roughly double the base hit) — a frozen ZINC encoder is worst
precisely on chemistry ZINC did not contain.

### Freezing the encoder does NOT disable flavour conditioning

Worth stating because it is a natural assumption and it is wrong.
`freeze=[encoder]` freezes `graph_encoder` alone. Measured on the frozen run's
checkpoint against its initialisation:

| tensor | mean abs delta | |
|---|---:|---|
| `graph_encoder.…embedding_atomic_num` | 0.000e+00 | frozen |
| `query_projector.projection.0` | 3.616e-01 | trained |
| `rgroup_projector.projection.0` | 2.839e-01 | trained |
| `node_projector.projection.0` | 4.177e-01 | trained |
| `assembly_head.fuse.0` | 2.239e-01 | trained |

And within the query projector, whose input is `300 node dims + 24 condvec`:

    node-dim columns [0:300]     mean abs delta 0.310
    CONDVEC columns  [300:324]   mean abs delta 1.004      3.2x more

The condvec columns move MORE, not less. With `q_i` fixed, the query projector
becomes the only place flavour information can enter, so it does more of the
work.

### But gradient reaching the heads is not the same as the heads being able

The heads train hard and still cannot compensate, because they can only compute
functions of a FIXED input. If the frozen embeddings do not encode a
distinction, no downstream projection creates it. The loss curves show that
ceiling directly:

| ep | trainable val/loss | frozen val/loss | trainable rgroup | frozen rgroup |
|---:|---:|---:|---:|---:|
| 0 | 4.8179 | 5.6350 | 4.2734 | 5.0376 |
| 4 | 4.2189 | 5.2293 | 3.7062 | 4.6326 |
| 8 | 4.0334 | 5.1507 | 3.5628 | 4.5394 |
| 12 | 3.9708 | **5.0925** | 3.4630 | 4.4869 |
| 16 | 3.9580 | 5.0983 | 3.4401 | 4.4817 |
| 19 | **3.9579** | 5.1017 | **3.4234** | **4.4912** |

Three things:

- the frozen run STARTS worse (5.635 vs 4.818) — the fixed embeddings are a
  worse substrate before any training happens;
- it improves by only 0.533 against 0.860, about 62% as much;
- it SATURATES at epoch 12 and then drifts slightly up, while the trainable run
  is still descending at epoch 19.

The R-group contrastive loss makes it plainest: frozen bottoms out at 4.49,
trainable reaches 3.42. That ~1.06 gap is not closed by eight further epochs of
projector training. The heads are optimising within a hypothesis space the
frozen encoder bounds.

So the 6.7-point retrieval drop is not a conditioning failure and not a lack of
gradient. It is a representation ceiling.

---

## 4. What ZINC actually bought: the assembly head

| | exact | control | head gap | aromaticity | atomic_num | total_num_hs |
|---|---:|---:|---:|---:|---:|---:|
| ZINC only (STEP 1) | 0.906 | 0.993 | 0.087 | 0.979 | 0.934 | 0.927 |
| ZINC → flavour | 0.966 | 0.990 | 0.023 | 0.997 | 0.977 | 0.979 |
| **ZINC → flavour, frozen** | **0.972** | 0.993 | **0.021** | 0.993 | 0.979 | 0.986 |

97.2% end-to-end molecule reconstruction WITH THE ENCODER FROZEN — the best of
any run. Assembly is chemistry-general in a way retrieval is not: predicting
bond order and atom identity at a cut point transfers across domains, while
ranking *which* substituent belongs is domain-specific.

No flavour-only checkpoint has a trained assembly head at all, so this
capability exists only because STEP 1 ran.

---

## 5. What this means for the three-step plan

**Step 1 earns its place, but for C, not A.** The original plan asked
`A (freeze?)` and `C (freeze?)`. The answers now differ:

- **Do NOT freeze the encoder in Step 2.** 20 SE of evidence. ZINC's encoder is
  a worse flavour encoder than one trained on flavour chemistry.
- **Freezing the assembly head is worth testing in Step 3.** It holds at 97.2%
  frozen, and 243 tastepocket records cannot teach assembly. Protecting it
  there is the same argument that justified training it on ZINC.

**The rising linker-contrastive loss is now confirmed across corpora.** It rose
through STEP 1 on ZINC (2.90 → 3.12) and rises in BOTH STEP 2 arms — trainable
2.75 → 3.39, frozen 2.97 → 3.57 — while total val loss falls in each. Three
corpora, two initialisations, same direction. That makes it a property of the
three-objective sum rather than of any dataset: `rgroup` improving pays for
`linker` degrading, and the summed loss hides the trade. It is the most
actionable open signal, and the cheapest test is to reweight the terms and see
whether retrieval moves.

## Reproducing

```bash
./run_experiment.sh step1                                   # ZINC pretraining
./run_experiment.sh step2                                   # flavour, all trainable
FREEZE=encoder ./run_experiment.sh step2                    # the ablation
CKPT=<checkpoint> ./run_experiment.sh assembly              # reconstruction
```

wandb: project `molplatte`, groups `step1-pretrain` and `step2-flavor-contrain`.
