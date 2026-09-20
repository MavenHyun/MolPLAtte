#!/usr/bin/env python3
"""Does pocket finetuning fail because pockets are useless, or because it forgets?

The September result -- pocket-trained 0.1119 vs no-finetune 0.1882 on unseen
receptors -- was read as "pocket conditioning is harmful". It cannot carry that
reading, because the run it came from froze only the graph encoder and left
**1.6M trainable parameters in the projectors** adapting to 269 records at
lr 1e-3, which is the sweep's WORST learning rate (0.3411 vs 0.3700 at 1e-4).
Catastrophic forgetting and "pockets do not help" predict the same number.

This separates them by construction rather than by inference:

  freeze=[encoder, projectors] leaves ONLY pocket_conditioning trainable --
  1,056 parameters behind a frozen PCA basis, with a zero-initialised output.
  The model therefore STARTS numerically identical to the no-finetune baseline
  and the flavour pathway CANNOT move. Any change is attributable to the pocket
  pathway alone. Forgetting is not measured; it is made impossible.

Arms (all 5-fold CV, held-out receptors, same w512 base checkpoint):

  enc-lr1e-3      freeze=[encoder]              lr 1e-3   the ORIGINAL run, reproduced
  enc-lr1e-4      freeze=[encoder]              lr 1e-4   isolates the learning rate
  adapter-lr1e-4  freeze=[encoder,projectors]   lr 1e-4   THE CLEAN TEST
  adapter-lr1e-3  freeze=[encoder,projectors]   lr 1e-3   freeze x lr interaction
  adapter-shuf    freeze=[encoder,projectors]   lr 1e-4   pocket half PERMUTED

The shuffled arm is the information control: identical capacity, identical
flavour bits, pocket content destroyed. adapter-lr1e-4 minus adapter-shuf is
the pocket's INFORMATION; anything else is capacity.

Baseline is RE-DERIVED here by scoring the base checkpoint through the same
eval path, never inherited from the 0.1882 in the notes.

Scoring is uniform: run_mode=test with cv_fold=k. val and test are the SAME
held-out fold (data_modules/base.py), so this scores exactly the held-out
receptors. That also means early stopping selected on the scored data -- an
optimistic bias shared by every arm INCLUDING the original, and one that
favours the high-capacity arms, so a win for the 1,056-param adapter is
conservative.

GPU1 ONLY -- GPU0 runs other jobs.
"""
from __future__ import annotations

import argparse, csv, os, re, statistics, subprocess, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
HOME = Path.home()
DATA = HOME / "preprocessed" / "molplatte"
CKPT = HOME / "checkpoints" / "molplatte"
LOGS = CKPT / "hplogs"
RESULTS = CKPT / "pocket_forgetting_results.csv"

BASE_CKPT = CKPT / "exp-wide512-pocket.pt"   # overridable via --base
CORPUS = "tastepocket_corpus"
VOCAB = DATA / "union_vocab" / "base-full__crossdocked__tastepocket" / "rgroup_vocab.pkl.gz"
BASIS = DATA / "pocket_basis"
EPOCHS = 40
SEED = 911012
FOLDS = [0, 1, 2, 3, 4]

# Shape-defining kwargs. Shared by TRAINING and SCORING, because a sweep stage
# once trained flavour-blind for 218 logged metrics before a shape mismatch
# exposed it. One list, used twice, checked after the fact.
def model_kwargs(fold: int) -> List[str]:
    return [
        "data_module_kwargs.condvec_dim=1304",
        "nnet_module_kwargs.condvec_dim=1304",
        "nnet_module_kwargs.hidden_dim=512",
        "nnet_module_kwargs.pocket_input_dim=1280",
        "nnet_module_kwargs.pocket_dim=32",
        "nnet_module_kwargs.pocket_dropout=0.0",
        f"+nnet_module_kwargs.pocket_basis_path={BASIS}/fold{fold}.npz",
    ]


@dataclass
class Arm:
    name: str
    freeze: List[str]
    lr: float
    shuffle_pocket: bool = False
    #: expected trainable parameter count, asserted against the log
    expect_trainable: Optional[int] = None


# A CAPACITY LADDER, each rung with a shuffled twin.
# The first pass tested only the extremes -- 1,056 params (inert) and 2.1M
# (destructive). The rungs between them are where a pocket could plausibly be
# used without collateral damage, and the query_projector is the specific place
# the pocket vector enters the model. Every rung gets a permuted-pocket twin so
# INFORMATION and CAPACITY stay separated at every capacity level; a rung that
# beats its own twin is the only thing that would count as pockets working.
_PROJ_NOT_QUERY = ["graph_projector", "node_projector", "rgroup_projector"]

ARMS = [
    Arm("pf-enc-lr1e-3",     ["encoder"],              1e-3),
    Arm("pf-enc-lr1e-4",     ["encoder"],              1e-4),
    Arm("pf-adapter-lr1e-4", ["encoder", "projectors"], 1e-4, expect_trainable=1056),
    Arm("pf-adapter-lr1e-3", ["encoder", "projectors"], 1e-3, expect_trainable=1056),
    Arm("pf-adapter-shuf",   ["encoder", "projectors"], 1e-4, shuffle_pocket=True,
        expect_trainable=1056),

    # rung: query_projector + pocket trainable (~556k) -- THE GAP
    Arm("pf-query-lr1e-4",      ["encoder"] + _PROJ_NOT_QUERY, 1e-4),
    Arm("pf-query-shuf",        ["encoder"] + _PROJ_NOT_QUERY, 1e-4, shuffle_pocket=True),
    # rung: nothing frozen at all (19.7M) -- the upper bound on capacity
    Arm("pf-full-lr1e-4",       [],                            1e-4),
    Arm("pf-full-shuf",         [],                            1e-4, shuffle_pocket=True),
    # twin for the 2.1M rung, which never had one
    Arm("pf-enc-lr1e-4-shuf",   ["encoder"],                   1e-4, shuffle_pocket=True),
]

RETR = re.compile(r"RGroupLibraryRetrieval/test\]")
FROZEN = re.compile(r"(FROZEN|trainable)\s*([a-z_]+)\s+([\d,]+) params")


def run(cmd: List[str], log: Path, env_extra: Dict[str, str]) -> int:
    env = dict(os.environ); env.update(env_extra)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        return subprocess.call(cmd, cwd=SRC, stdout=fh, stderr=subprocess.STDOUT, env=env)


def trainable_from_log(log: Path) -> Optional[int]:
    """Parameters the run actually left trainable, read back from its own log.

    A freeze that silently did nothing would leave the adapter arms identical to
    the encoder arms and the whole experiment would answer the wrong question.
    """
    if not log.exists():
        return None
    total = None
    for line in log.read_text(errors="ignore").splitlines():
        m = FROZEN.search(line)
        if m and m.group(1) == "trainable":
            total = (total or 0) + int(m.group(3).replace(",", ""))
    return total


def train(arm: Arm, fold: int, gpu: str, dry: bool, batch_size=None) -> str:
    exp = f"{arm.name}-f{fold}"
    log = LOGS / f"{exp}.log"
    if log.exists():
        print(f"  SKIP  {exp} (log exists)")
        return exp
    cmd = [sys.executable, "-u", "run.py",
           f"experiment_name={exp}",
           f"hydra.run.dir={REPO}/outputs/{exp}",
           f"random_seed={SEED}",
           f"init_weights_from={BASE_CKPT}",
           f"data_module_kwargs.dataset_version={CORPUS}",
           f"+data_module_kwargs.cv_fold={fold}",
           f"lightning_module_kwargs.learning_rate={arm.lr}",
           *([] if not arm.freeze else [f"freeze=[{','.join(arm.freeze)}]"]),
           f"++trainer_kwargs.max_epochs={EPOCHS}",
           *([] if batch_size is None else [f"data_module_kwargs.batch_size={batch_size}"]),
           f"rgroup_library.vocab_path={VOCAB}",
           "rgroup_library.enable_after_epoch=0",
           "wandb.project=null"] + model_kwargs(fold)
    if arm.shuffle_pocket:
        cmd.append("data_module_kwargs.shuffle_pocket_only=true")
    if dry:
        print("   ", " ".join(cmd[2:])); return exp
    t = time.time()
    rc = run(cmd, log, {"CUDA_VISIBLE_DEVICES": gpu, "WANDB_MODE": "disabled"})
    print(f"  {exp:<28} rc={rc}  {(time.time()-t)/60:.1f} min")
    return exp


def score(ckpt: Path, fold: int, tag: str, gpu: str,
          shuffle_pocket: bool = False, dry: bool = False) -> Dict:
    """THE one scoring path -- identical for every arm and for the baseline."""
    log = LOGS / f"eval-{tag}.log"
    if not log.exists() and not dry:
        cmd = [sys.executable, "-u", "run.py", "run_mode=test",
               f"experiment_name=eval-{tag}",
               f"hydra.run.dir=/tmp/hyd-eval-{tag}",
               f"random_seed={SEED}",
               f"+checkpoint_path={ckpt}",
               f"data_module_kwargs.dataset_version={CORPUS}",
               f"+data_module_kwargs.cv_fold={fold}",
               f"rgroup_library.vocab_path={VOCAB}",
               "rgroup_library.enable_after_epoch=0",
               "wandb.project=null"] + model_kwargs(fold)
        if shuffle_pocket:
            cmd.append("data_module_kwargs.shuffle_pocket_only=true")
        run(cmd, log, {"CUDA_VISIBLE_DEVICES": gpu, "WANDB_MODE": "disabled"})
    if dry or not log.exists():
        return {}
    lines = [l for l in log.read_text(errors="ignore").splitlines() if RETR.search(l)]
    if not lines:
        return {"error": "no retrieval line"}
    s = lines[-1]
    g = lambda k: (float(m.group(1)) if (m := re.search(rf"{k}=([0-9.]+)", s)) else None)
    pm = re.search(r"H@1=[0-9.]+\(prior ([0-9.]+)\)", s)
    return {"H@1": g("H@1"), "H@10": g("H@10"), "MRR": g("MRR"),
            "novel@10": g("novel_hit@10"),
            "prior@1": float(pm.group(1)) if pm else None,
            "N": g("N")}


def mean_se(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = statistics.fmean(xs)
    se = (statistics.stdev(xs) / len(xs) ** 0.5) if len(xs) > 1 else 0.0
    return m, se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--base", default=None, help="override the init checkpoint")
    ap.add_argument("--prefix", default=None, help="rename arms, e.g. pfr- for the rescaled ladder")
    ap.add_argument("--results", default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="training batch size. The default 512 gives ONE batch per "
                         "epoch on this corpus (~20 gradient steps per run); a smaller "
                         "value trades in-batch negatives for optimisation steps.")
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(",")]
    global BASE_CKPT, RESULTS
    if a.base:
        BASE_CKPT = Path(a.base)
    if a.results:
        RESULTS = Path(a.results)
    # The baseline tags MUST carry the prefix too. Without it a rerun on new
    # fold definitions finds the previous run's eval logs, skips re-scoring, and
    # silently compares new arms against a baseline computed on OLD folds.
    tag_prefix = "pf-"
    if a.prefix:
        tag_prefix = a.prefix
        for arm in ARMS:
            arm.name = arm.name.replace("pf-", a.prefix, 1)
    assert BASE_CKPT.exists(), f"no base checkpoint {BASE_CKPT}"

    rows = []

    print("== baseline: the base checkpoint, no finetuning at all")
    base = {}
    for k in folds:
        r = score(BASE_CKPT, k, f"{tag_prefix}baseline-f{k}", a.gpu, dry=a.dry)
        base[k] = r.get("H@1")
        print(f"  fold {k}: H@1={r.get('H@1')}  N={r.get('N')}")
        rows.append(("baseline", k, r, None))

    # the same base checkpoint with the pocket destroyed -- if this differs from
    # the line above, the UNTRAINED pocket path is already doing something
    print("== baseline with pocket permuted (zero-init sanity check)")
    for k in folds:
        r = score(BASE_CKPT, k, f"{tag_prefix}baseline-shuf-f{k}", a.gpu,
                  shuffle_pocket=True, dry=a.dry)
        print(f"  fold {k}: H@1={r.get('H@1')}")
        rows.append(("baseline-shuf", k, r, None))

    for arm in ARMS:
        print(f"\n== {arm.name}   freeze={arm.freeze} lr={arm.lr}"
              f"{' SHUFFLED-POCKET' if arm.shuffle_pocket else ''}")
        for k in folds:
            exp = train(arm, k, a.gpu, a.dry, a.batch_size)
            if a.dry:
                continue
            ntr = trainable_from_log(LOGS / f"{exp}.log")
            if arm.expect_trainable is not None and ntr is not None:
                assert ntr == arm.expect_trainable, (
                    f"{exp}: freeze did not take effect -- {ntr:,} trainable params, "
                    f"expected {arm.expect_trainable:,}")
            ck = CKPT / f"{exp}_best.pt"
            if not ck.exists():
                print(f"  {exp}: NO CHECKPOINT"); rows.append((arm.name, k, {"error": "no ckpt"}, ntr)); continue
            r = score(ck, k, f"{exp}", a.gpu, shuffle_pocket=arm.shuffle_pocket)
            print(f"  fold {k}: H@1={r.get('H@1')}  trainable={ntr:,}" if ntr else
                  f"  fold {k}: H@1={r.get('H@1')}")
            rows.append((arm.name, k, r, ntr))

    if a.dry:
        return 0

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "fold", "H@1", "H@10", "MRR", "novel@10", "prior@1", "N", "trainable"])
        for name, k, r, ntr in rows:
            w.writerow([name, k, r.get("H@1"), r.get("H@10"), r.get("MRR"),
                        r.get("novel@10"), r.get("prior@1"), r.get("N"), ntr])

    print("\n" + "=" * 78)
    print(f"{'arm':<24}{'H@1 mean':>10}{'SE':>8}{'vs base':>10}{'folds better':>14}")
    bm, _ = mean_se(list(base.values()))
    by = {}
    for name, k, r, _ in rows:
        by.setdefault(name, {})[k] = r.get("H@1")
    for name, d in by.items():
        m, se = mean_se(list(d.values()))
        if m is None:
            continue
        better = sum(1 for k, v in d.items()
                     if v is not None and base.get(k) is not None and v > base[k])
        delta = "" if name.startswith("baseline") else f"{m - bm:+.4f}"
        print(f"{name:<24}{m:>10.4f}{se:>8.4f}{delta:>10}{better:>10}/{len(d)}")
    print(f"\nbaseline (no finetune) = {bm:.4f}")
    print(f"results -> {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
