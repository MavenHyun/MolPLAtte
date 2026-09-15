#!/usr/bin/env python3
"""Staged hyperparameter sweep across s1 -> s2 -> s3.

STAGED, NOT FULL-FACTORIAL. s1 costs ~2 h and s2 ~30 min, so a full grid would
spend most of its budget on the expensive stage. Instead: a handful of s1
variants, then the cheap s2 grid on the BEST s1 only, then the winning s2
config applied to the remaining s1 checkpoints.

Three axes are deliberately absent because they are already measured null, and
including them would spend the budget confirming nothing:

    linker loss weight   43% swing in its own loss, retrieval moved < 1 SE
    pocket capacity      null at 0, 1,056 and 168,096 parameters
    s3 finetune depth    finetuning on tastepocket measured WORSE than not

WHAT THIS FILE IS REALLY FOR is the scoring discipline. Every arm is scored by
ONE function, with the same split, the same library and the same query
subsample, and the baseline is re-run INSIDE the sweep rather than compared to
a historical number. Both of today's invalidated comparisons came from
violating exactly that: a test-split baseline held up against val-split arms,
and a warm start that silently differed between two runs. A sweep is a
comparison machine; if its comparisons are not like-for-like it is a random
number generator with a progress bar.

Resumable: an arm whose log already exists is skipped, so the sweep can be
killed and restarted without losing work or double-counting.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
CKPT = Path.home() / "checkpoints" / "molplatte"
LOGS = CKPT / "hplogs"
DATA = Path.home() / "preprocessed" / "molplatte"

#: The ONE evaluation. Every arm is scored against this library on this corpus.
EVAL_CORPUS = "coconut-flavordb-full"
EVAL_VOCAB = DATA / "union_vocab" / "zincbase__flavor__crossdocked__tastepocket" / "rgroup_vocab.pkl.gz"
EVAL_SEED = 911012

RESULTS = CKPT / "sweep_results.csv"


#: Overrides that change the MODEL'S SHAPE. They must be identical in training
#: and in scoring, so they live in one place and both paths read them.
#: Getting this wrong is silent: the first run of this sweep trained four arms
#: at condvec_dim=0 -- flavour-blind, and not comparable to the baseline's 24 --
#: because the training command omitted it and inherited the config default.
#: All four returned rc=0, wrote checkpoints and logged 218 metrics each. Only
#: the scoring shape-mismatch exposed it, and only because scoring did NOT
#: infer its shape from the checkpoint.
BASE_MODEL: List[str] = [
    "data_module_kwargs.condvec_dim=24",
    "nnet_module_kwargs.condvec_dim=24",
]


@dataclass
class Arm:
    name: str
    stage: str                      # s1 | s2 | s3
    corpus: str
    epochs: int
    #: Shape-affecting; applied to training AND scoring.
    model: List[str] = field(default_factory=lambda: list(BASE_MODEL))
    #: Training-only (schedule, freezing, loss weights).
    overrides: List[str] = field(default_factory=list)
    init_from: Optional[str] = None      # checkpoint stem to warm start from
    seed: int = EVAL_SEED

    @property
    def log(self) -> Path:
        return LOGS / f"{self.name}.log"

    @property
    def ckpt(self) -> Path:
        return CKPT / f"{self.name}_best.pt"


def run(cmd: List[str], log: Path, env: Optional[dict] = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ)
    e.update(env or {})
    with open(log, "w") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=SRC, env=e).returncode


def assert_trained_as_scored(arm: Arm) -> Optional[str]:
    """Did the run actually use the shape we will score it with?

    The resolved Hydra config is the ground truth -- an override that was
    dropped, misspelled or shadowed by a default does not appear there, and the
    run succeeds regardless. Checked AFTER training and BEFORE scoring, because
    a mismatch means the arm answers a different question than the one asked.
    """
    cfg = REPO / "outputs" / arm.name / ".hydra" / "config.yaml"
    if not cfg.is_file():
        return f"no resolved config at {cfg}"
    # Resolve the FULL dotted path, not the leaf name. Four blocks in this
    # config define `temperature` (graph 0.1, linker 0.05, rgroup 0.01,
    # assembly 0.1), so a leaf-name regex matches whichever appears first and
    # compares the wrong one -- which failed all eight Stage B arms whose
    # training was in fact correct.
    from omegaconf import OmegaConf

    conf = OmegaConf.load(cfg)
    for item in arm.model:
        key, _, want = item.partition("=")
        # `+key=value` and `key=value` name the same node.
        path = key.lstrip("+")
        got = OmegaConf.select(conf, path, default=None)
        if got is None:
            return f"{path} absent from the resolved config"
        if str(got).strip().lower() != want.strip().lower():
            return f"{path}: trained with {got}, scoring wants {want}"
    return None


def train(arm: Arm, gpu: str, dry: bool = False) -> bool:
    if arm.log.exists():
        print(f"  SKIP {arm.name} (log exists)")
        return arm.ckpt.exists()
    cmd = [sys.executable, "-u", "run.py",
           f"experiment_name={arm.name}",
           f"hydra.run.dir={REPO}/outputs/{arm.name}",
           f"random_seed={arm.seed}",
           f"data_module_kwargs.dataset_version={arm.corpus}",
           f"++trainer_kwargs.max_epochs={arm.epochs}",
           f"rgroup_library.vocab_path={EVAL_VOCAB}",
           "wandb.project=molplatte", f"wandb.group=sweep-{arm.stage}",
           f"wandb.name={arm.name}", f"wandb.job_type={arm.stage}"]
    if arm.init_from:
        cmd.append(f"init_weights_from={CKPT / arm.init_from}")
    cmd += arm.model + arm.overrides
    if dry:
        print("    would run:", " ".join(cmd[2:]))
        return True
    t0 = time.time()
    rc = run(cmd, arm.log, {"CUDA_VISIBLE_DEVICES": gpu})
    print(f"  {arm.name:<38} rc={rc}  {(time.time()-t0)/60:.0f} min")
    return rc == 0 and arm.ckpt.exists()


def score(arm: Arm, gpu: str, dry: bool = False) -> Dict:
    """THE one scoring path. Identical for every arm, no exceptions."""
    out = LOGS / f"eval-{arm.name}.log"
    if not out.exists() and not dry:
        cmd = [sys.executable, "-u", "run.py", "run_mode=test",
               f"experiment_name=eval-{arm.name}",
               f"hydra.run.dir=/tmp/hyd-eval-{arm.name}",
               f"random_seed={EVAL_SEED}",
               f"+checkpoint_path={arm.ckpt}",
               f"data_module_kwargs.dataset_version={EVAL_CORPUS}",
               f"rgroup_library.vocab_path={EVAL_VOCAB}",
               "rgroup_library.enable_after_epoch=0"] + arm.model
        run(cmd, out, {"CUDA_VISIBLE_DEVICES": gpu, "WANDB_MODE": "disabled"})
    if dry or not out.exists():
        return {}
    txt = out.read_text(errors="ignore")
    line = [l for l in txt.splitlines() if "RGroupLibraryRetrieval/test]" in l]
    if not line:
        return {"error": "no retrieval line"}
    s = line[-1]
    g = lambda k: (float(m.group(1))
                   if (m := re.search(rf"{k}=([0-9.]+)", s)) else None)
    # The prior is parenthesised, not "key=value" -- the generic pattern above
    # cannot reach it and silently returned None.
    pm = re.search(r"H@1=[0-9.]+\(prior ([0-9.]+)\)", s)
    return {"H@1": g("H@1"), "H@10": g("H@10"), "H@100": g("H@100"),
            "MRR": g("MRR"),
            "prior@1": float(pm.group(1)) if pm else None,
            "novel@10": g("novel_hit@10")}


def record(arm: Arm, res: Dict, note: str = "", dry: bool = False) -> None:
    if dry:
        return
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["name", "stage", "corpus", "epochs", "seed",
                        "init_from", "overrides", "H@1", "H@10", "H@100",
                        "MRR", "prior@1", "novel@10", "note"])
        w.writerow([arm.name, arm.stage, arm.corpus, arm.epochs, arm.seed,
                    arm.init_from or "", " ".join(arm.overrides),
                    res.get("H@1"), res.get("H@10"), res.get("H@100"),
                    res.get("MRR"), res.get("prior@1"), res.get("novel@10"),
                    note or res.get("error", "")])


# ---------------------------------------------------------------- stage A
# s1 variants. ~2 h each, so only the axes that could plausibly matter.
# Baselines are REUSED where an identical run already exists -- but they are
# re-SCORED through score() below, never compared to a historical number.
S1_ARMS = [
    # The one that could retire a whole stage. ZINC's only surviving
    # justification is that it delivers the assembly head (it costs -0.0125 H@1
    # on retrieval). If flavour-only produces a working head, ZINC has no role.
    Arm("sw-s1-flavour-assembly", "s1", "coconut-flavordb-full", 30,
        model=BASE_MODEL + ["assembly.enabled=true"]),
    # Depth. The MAD curve declines 33% over 5 conv layers (41% on ZINC), so
    # shallower may over-smooth less -- the one architectural lead the health
    # measurements produced.
    Arm("sw-s1-conv3", "s1", "coconut-flavordb-full", 30,
        model=BASE_MODEL + ["nnet_module_kwargs.num_conv=3"]),
    Arm("sw-s1-conv7", "s1", "coconut-flavordb-full", 30,
        model=BASE_MODEL + ["nnet_module_kwargs.num_conv=7"]),
    # Width.
    Arm("sw-s1-wide512", "s1", "coconut-flavordb-full", 30,
        model=BASE_MODEL + ["nnet_module_kwargs.hidden_dim=512"]),
]

# ---------------------------------------------------------------- stage W
# Width scan. hidden_dim 512 beat the 300 baseline by +0.0367 H@1 (+11 SE), the
# largest effect this project has measured -- but at 19.0M parameters against
# 6.6M, so "wider helps" and "more capacity helps" are not yet separable, and a
# single point cannot tell a trend from a lucky draw. These fill in the curve.
#
# Reported WITH parameter counts, because the practical question is not just
# whether H@1 rises but whether it is worth 3x the inference cost.
S1_WIDTH = [
    Arm(f"sw-s1-w{w}", "s1", "coconut-flavordb-full", 30,
        model=BASE_MODEL + [f"nnet_module_kwargs.hidden_dim={w}"])
    for w in (384, 768, 1024)
]

#: Already trained, identical config -- reused rather than retrained, and
#: re-scored through the same path as everything else.
S1_EXISTING = {
    "sw-s1-baseline": "s1-pretrain-full-v3-s911012",
}

# ---------------------------------------------------------------- stage B
# The cheap stage, swept in full on the BEST s1 only. tau is the headline: it
# is the ranking objective itself and has never been varied in training.
#: s1 the grid branches from. Width 512 is the knee of the scaling curve:
#: +0.0367 H@1 over the 300 baseline for 2.9x the parameters, capturing 63% of
#: the gain that 1024 buys for 11.5x.
S2_BASE = "sw-s1-wide512"
S2_SHAPE = BASE_MODEL + ["nnet_module_kwargs.hidden_dim=512"]

# tau goes in the SHAPE block, not the training block. The corrected score is
# sim/tau + coef*log p(k), so a model trained at 0.05 must be SCORED at 0.05 --
# scoring every arm at the config default would have made this sweep's headline
# axis meaningless while still producing a tidy table.
S2_GRID = [
    (f"fr{fr}-tau{tau}-lr{lr}",
     S2_SHAPE + [f"loss_module_kwargs.loss_rgroup_kwargs.temperature={tau}"],
     ([] if fr == "none" else [f"freeze=[{fr}]"])
     + [f"lightning_module_kwargs.learning_rate={lr}"])
    for fr in ("none", "encoder")
    for tau in (0.01, 0.05)
    for lr in ("1.0e-3", "3.0e-4")
]

S2_ARMS = [
    Arm(f"sw-s2-{tag}", "s2", "coconut-flavordb-full", 20,
        model=shape, overrides=train, init_from=f"{S2_BASE}_best.pt")
    for tag, shape, train in S2_GRID
]

# ---------------------------------------------------------------- stage B2
# lr was the dominant axis of the grid (+0.0109 for 3e-4 over 1e-3) and 3e-4
# sits at the EDGE of it, with the trend not yet turned over -- so the grid
# found a boundary, not an optimum. tau is pinned at 0.01, which the grid
# confirmed as already correct. Both freeze settings are carried at 1e-4
# because a freeze x lr interaction cannot be ruled out from marginal means
# alone, even though the freeze axis itself was null (+0.0005).
S2_LR = [
    Arm(f"sw-s2-fr{fr}-tau0.01-lr{lr}", "s2", "coconut-flavordb-full", 20,
        model=S2_SHAPE + ["loss_module_kwargs.loss_rgroup_kwargs.temperature=0.01"],
        overrides=([] if fr == "none" else [f"freeze=[{fr}]"])
                  + [f"lightning_module_kwargs.learning_rate={lr}"],
        init_from=f"{S2_BASE}_best.pt")
    for fr, lr in (("none", "1.0e-4"), ("none", "3.0e-5"), ("encoder", "1.0e-4"))
]

# ---------------------------------------------------------------- stage D
# Seed replication. Every number up to here is ONE seed, and the top arms span
# ~3 SE -- enough to rank the AXES, not enough to crown a CELL. Two extra seeds
# on the three best lr settings, plus the freeze=encoder arm at the winning lr,
# because the freeze effect REVERSED with lr (helps at 1e-3, hurts at 1e-4) and
# a reversal inferred from single runs is exactly the claim a second seed
# either confirms or dissolves.
#
# LIMITATION, stated because it bounds what this can conclude: only the s2 seed
# varies. All arms warm-start from the same single-seed sw-s1-wide512, so this
# measures s2 variance, not end-to-end pipeline variance. Re-seeding s1 costs
# ~1.6 h per seed and is a separate question.
SEEDS_EXTRA = (20260914, 4242)
S2_SEED_CONFIGS = [("none", "1.0e-4"), ("none", "3.0e-4"),
                   ("none", "3.0e-5"), ("encoder", "1.0e-4")]
S2_SEEDS = [
    Arm(f"sw-s2-fr{fr}-tau0.01-lr{lr}-s{sd}", "s2", "coconut-flavordb-full", 20,
        model=S2_SHAPE + ["loss_module_kwargs.loss_rgroup_kwargs.temperature=0.01"],
        overrides=([] if fr == "none" else [f"freeze=[{fr}]"])
                  + [f"lightning_module_kwargs.learning_rate={lr}"],
        init_from=f"{S2_BASE}_best.pt", seed=sd)
    for fr, lr in S2_SEED_CONFIGS for sd in SEEDS_EXTRA
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["A", "W", "B1", "B2", "BLR", "C", "D", "all"],
                    default="A")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"sweep stage {args.stage}  gpu {args.gpu}"
          f"{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"scoring: corpus={EVAL_CORPUS} vocab={EVAL_VOCAB.parent.name} "
          f"seed={EVAL_SEED}\n")

    if args.stage in ("W", "all"):
        for arm in S1_WIDTH:
            ok = train(arm, args.gpu, args.dry_run)
            if not ok and not args.dry_run:
                record(arm, {}, "TRAIN FAILED", args.dry_run)
                continue
            bad = None if args.dry_run else assert_trained_as_scored(arm)
            if bad:
                print(f"  {arm.name:<38} CONFIG MISMATCH: {bad}")
                record(arm, {}, f"CONFIG MISMATCH: {bad}", args.dry_run)
                continue
            res = score(arm, args.gpu, args.dry_run)
            record(arm, res, "", args.dry_run)
            print(f"  {arm.name:<38} H@1={res.get('H@1')}")

    if args.stage in ("D", "all"):
        for arm in S2_SEEDS:
            ok = train(arm, args.gpu, args.dry_run)
            if not ok and not args.dry_run:
                record(arm, {}, "TRAIN FAILED", args.dry_run); continue
            bad = None if args.dry_run else assert_trained_as_scored(arm)
            if bad:
                print(f"  {arm.name:<44} CONFIG MISMATCH: {bad}")
                record(arm, {}, f"CONFIG MISMATCH: {bad}", args.dry_run); continue
            res = score(arm, args.gpu, args.dry_run)
            record(arm, res, "", args.dry_run)
            print(f"  {arm.name:<44} H@1={res.get('H@1')}")

    if args.stage in ("BLR", "all"):
        for arm in S2_LR:
            ok = train(arm, args.gpu, args.dry_run)
            if not ok and not args.dry_run:
                record(arm, {}, "TRAIN FAILED", args.dry_run); continue
            bad = None if args.dry_run else assert_trained_as_scored(arm)
            if bad:
                print(f"  {arm.name:<38} CONFIG MISMATCH: {bad}")
                record(arm, {}, f"CONFIG MISMATCH: {bad}", args.dry_run); continue
            res = score(arm, args.gpu, args.dry_run)
            record(arm, res, "", args.dry_run)
            print(f"  {arm.name:<38} H@1={res.get('H@1')}")

    if args.stage in ("B1", "all"):
        for arm in S2_ARMS:
            ok = train(arm, args.gpu, args.dry_run)
            if not ok and not args.dry_run:
                record(arm, {}, "TRAIN FAILED", args.dry_run)
                continue
            bad = None if args.dry_run else assert_trained_as_scored(arm)
            if bad:
                print(f"  {arm.name:<38} CONFIG MISMATCH: {bad}")
                record(arm, {}, f"CONFIG MISMATCH: {bad}", args.dry_run)
                continue
            res = score(arm, args.gpu, args.dry_run)
            record(arm, res, "", args.dry_run)
            print(f"  {arm.name:<38} H@1={res.get('H@1')}")

    if args.stage in ("A", "all"):
        for _, existing in S1_EXISTING.items():
            # Name the Arm after the checkpoint that exists, so .ckpt resolves
            # without patching the object afterwards.
            arm = Arm(existing, "s1", "coconut-flavordb-full", 30)
            if not arm.ckpt.exists():
                print(f"  MISSING baseline {arm.ckpt}")
                continue
            res = score(arm, args.gpu, args.dry_run)
            record(arm, res, "reused baseline, re-scored here", args.dry_run)
            print(f"  {existing:<38} H@1={res.get('H@1')}")
        for arm in S1_ARMS:
            ok = train(arm, args.gpu, args.dry_run)
            if not ok and not args.dry_run:
                record(arm, {}, "TRAIN FAILED", args.dry_run)
                continue
            bad = None if args.dry_run else assert_trained_as_scored(arm)
            if bad:
                print(f"  {arm.name:<38} CONFIG MISMATCH: {bad}")
                record(arm, {}, f"CONFIG MISMATCH: {bad}", args.dry_run)
                continue
            res = score(arm, args.gpu, args.dry_run)
            record(arm, res, "", args.dry_run)
            print(f"  {arm.name:<38} H@1={res.get('H@1')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
