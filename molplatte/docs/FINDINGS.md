# MolPLAtte: what we built, what works, and why pockets don't

Plain-language summary. Every number here is reproducible from the scripts in
`molplatte/src/scripts/`; the per-experiment documents are indexed in
[README.md](README.md).

---

## In one paragraph

MolPLAtte suggests chemical modifications to flavour compounds: give it a
molecule and a target taste, and it proposes replacement fragments. Tuning the
model properly improved it **19%** (H@1 0.3098 → 0.3701), and that model is
shipped. Conditioning it on the **protein binding pocket** does not work -- but
not for the reason we assumed for most of this project. The pocket genuinely
does carry information about the right answer. The problem is that the model
already had that information from somewhere cheaper.

---

## Part 1 — What the model does

A molecule is cut into a **core** and one or more **R-groups** (decorations
hanging off the core). The model sees the core with a gap and picks the missing
R-group from a catalogue of **91,935** fragments.

It scores candidates with:

```
score = (how well the fragment fits) / temperature  +  log(how common the fragment is)
```

That second term matters enormously for everything below. **The frequency of
each fragment is already part of the answer.** It is not an add-on; it is in the
objective the model is trained with.

---

## Part 2 — What worked

| change | effect |
|---|---|
| width 512, learning rate 1e-4 | **H@1 0.3098 → 0.3701 (+19%)**, novel@10 +25% |
| dropping ZINC pretraining | *improves* retrieval by 0.0125 |
| freezing the encoder | **hurts** at the tuned learning rate (−0.0103) |

The last two are worth pausing on: both were previously believed to help. The
freezing "gain" was an artifact of an untuned learning rate -- at lr 1e-3
freezing helps, at lr 1e-4 it hurts. Sweeping the learning rate *first* changed
the sign of the conclusion.

---

## Part 3 — Why the pocket doesn't help

This is the part worth reading carefully, because "we don't have enough data" is
only half of it.

### The wrong question and the right one

**Wrong question:** *does the binding pocket know anything about which fragment
belongs there?*

It does. We measured it directly. Take a pocket, find the most similar *other*
pocket (different protein, different ligand), and look at what fragments its
ligand carries: they match the true answer far more often than chance, at
**20.8 sigma**. The pocket even predicts fragments slightly *better* than it
predicts whole molecules.

**Right question:** *does the pocket know anything the frequency table doesn't
already know?*

Barely. And that is the whole story.

### Why the overlap is so large

Flavour molecules are small and decorated from a shared, narrow vocabulary:
hydroxyls, methyls, short chains, simple rings. Concretely:

- the typical fragment our taste pockets need appears **~1,200 times** in a
  393,000-molecule reference corpus
- those fragments are **0.3% of the catalogue** but carry **49% of all fragment
  occurrences**

So "just guess a common fragment" is already a strong strategy — and the model
is already doing exactly that, via the `log(how common)` term.

### An analogy

Imagine guessing which letters appear in a word. Knowing the topic helps a
little. But *e*, *t*, *a*, *o* appear in almost every word regardless, so
knowing the topic adds little over just knowing letter frequencies — **unless
you only get one guess**, in which case the topic genuinely helps.

That is exactly the shape of what we measured.

### The measurement

Predicting a query's fragments, at equal budget *k*:

| how many guesses | frequency alone | nearest pocket | winner |
|---:|---:|---:|---|
| 1 | 0.0436 | **0.1174** | pocket, **2.7x** |
| 5 | 0.1455 | **0.1756** | pocket, modest |
| 10 | 0.2561 | 0.2396 | tie |
| 100 | **0.5703** | 0.3744 | **frequency, by a lot** |

The pocket is a **sharp but shallow** signal. It is genuinely good at picking
the single best answer and useless for building a long list. (The reason is
structural: each ligand carries only ~1.2 fragments, so extending a
pocket-based list means reaching for ever more distant pockets, while the
frequency list just keeps covering the common head.)

### Why training can never find it

The thing to be learned is worth roughly **+0.03** in hit rate. We have **269**
training records. Run-to-run variation is of comparable size. The optimiser is
chasing a signal about as large as its own noise.

And it demonstrably never finds it. We trained one model on **real** pockets and
an identical model on **randomly shuffled** pockets:

| fold | real pockets | shuffled pockets | difference |
|---|---:|---:|---:|
| 0 | 8.4352 | 8.4356 | 0.0004 |
| 1 | 7.4285 | 7.4284 | 0.0001 |
| 2 | 8.1587 | 8.1583 | 0.0004 |
| 3 | 7.3938 | 7.3940 | 0.0002 |
| 4 | 7.6398 | 7.6400 | 0.0002 |

Identical to four decimal places. **The optimiser cannot tell a real pocket from
a scrambled one.** We repeated this at four model sizes (1,056 to 19.7M
trainable parameters), two learning rates, and two initialisation scales --
eight comparisons, all null.

### So: is it just the dataset?

**No — it is the product of two things, and either one alone would have been
survivable.**

- a **large** effect would be learnable from 269 records
- a **small** effect would be learnable from 100,000 records
- a **small effect and a small dataset** leaves nothing to learn

The effect is small *because the frequency prior already covers most of it* —
that part is a property of flavour chemistry, not of our data collection. More
tastepocket complexes would shrink the noise but not grow the signal.

### What this is NOT

Three explanations we tested and ruled out, each of which we believed at some
point:

- **Not bad pocket extraction.** 1,247 of 1,254 pockets make chemically
  coherent contacts with their own crystal ligand (mean 36.5 contacts).
- **Not the wrong pocket representation.** 3D geometry and typed
  interaction fingerprints were both tested; neither beats the sequence
  embedding, and a *random-weight* 3D network gets within 90% of it.
- **Not the retrieval target being too coarse.** We believed this for most of
  the project and wrote it into three documents. It is false: pockets predict
  fragments at 20.8 sigma.

---

## Part 4 — What the pocket IS good for

The receptor structure still earns its place in the pipeline, just not as model
conditioning:

- **Docking.** Every proposed compound is docked into the real receptor with
  AutoDock Vina, giving a binding score completely independent of the model —
  free to disagree with it. The crystal ligand is redocked first as a control
  (under ~2 Å means the setup reproduces a known answer).
- **Pose rendering.** The 3D pictures in the report come from real structures.
- **A top-5 re-ranking prior.** Worth adding, worth capping. Past ~10
  candidates it is worse than guessing common fragments.

---

## Part 5 — What we got wrong along the way

Recorded because the corrections are part of the result:

1. **"Pocket conditioning is harmful" (−0.0763).** Wrong attribution. That run
   left 2.1M parameters adapting to 269 records at the worst learning rate in
   the sweep, so it measured **catastrophic forgetting**, not pockets. Freeze
   the flavour pathway and the loss vanishes completely.
2. **"The retrieval target is too coarse."** Refuted at 20.8 sigma. A plausible
   story that fit every result, repeated across three documents, never measured
   until it was.
3. **"Nearest-pocket lookup gives 15x enrichment."** True but misleading — it
   was measured against a random baseline rather than against the frequency
   prior, which is the competitor that matters.
4. **A proposed fix that was a no-op.** We rescaled the pocket weights 40x to
   test whether they were initialised too weakly. They were — but the model
   trains with Adam, which normalises away gradient scale, so the intervention
   changed nothing (1.0x on every rung).

---

## Part 6 — Honest limitations

- **Single seed** (911012) across all pocket experiments; the ±0.003 spreads
  called "noise" were never checked against seed variance.
- **`val` and `test` are the same held-out fold**, so early stopping selected on
  the scored data. Shared by every arm, so comparisons are fair, but absolute
  numbers are optimistic.
- **17 receptor families, 96 receptors**, and one family (TRPV1) is 522 of
  1,255 pockets. Cross-family claims rest on few families.

---

## Part 7 — What to do next

1. **Ship what works.** `molplatte-final-v2-w512.pt` is the tuned model; the
   notebook uses it.
2. **Add the pocket as a top-5 prior, not a conditioning input.** It needs no
   training and is 2.7x better than frequency at the top slot.
3. **Retire the ZINC stage.** It costs 0.0125 H@1 and its last justification is
   matched by flavour-only training.
4. **Don't collect more tastepocket data expecting conditioning to start
   working.** More records shrink the noise; they do not grow the signal.
