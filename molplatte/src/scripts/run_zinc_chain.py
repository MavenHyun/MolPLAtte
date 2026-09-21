#!/usr/bin/env python3
"""Does ZINC pretraining help, measured on a split that does not leak?

The 2026-09-09 verdict (-0.0125 H@1, "retire the ZINC stage") was taken at
hidden_dim=300 / lr=1e-3, SIX DAYS before the sweep that found width and
learning rate were worth +19%. That is the same setup in which freezing
"helped" and then reversed sign once the learning rate was tuned, so the ZINC
result was never safe.

It also predates two leaks found on 2026-09-21:
  * steps 1-2 split by DECOMPOSITION, so 91.9% of eval items came from
    molecules also present in training;
  * the R-group library counted ALL molecules, putting the test set into the
    log p(k) prior that is half the scoring function.

Both are fixed here: split_by=molecule (stable hash of the id, so a molecule
lands in the same split in every corpus containing it) and a train-only
vocabulary.

FOUR runs, two arms:

  arm Z   s1 on zinc2020+coconut-flavordb  ->  s2 on flavour  ->  s3 on pockets
  arm F   s1 on coconut-flavordb only      ->  s2 on flavour  ->  s3 on pockets

Arm F is not the old shipped checkpoint: it is a fresh control on the SAME
fixed split, because comparing the new chain against the historical 0.3701
would confound ZINC with the split fix.

GPU1 ONLY -- GPU0 runs other jobs.
"""
from __future__ import annotations
import argparse, csv, json, os, re, subprocess, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
HOME = Path.home()
DATA = HOME / "preprocessed" / "molplatte"
CKPT = HOME / "checkpoints" / "molplatte"
LOGS = CKPT / "hplogs"
RESULTS = CKPT / "zinc_chain_results.csv"

MERGED = "zinc2020-coconut-flavordb-full"
FLAVOUR = "coconut-flavordb-full"
POCKETS = "tastepocket_corpus"
EXCLUDE = DATA / "exclude_tastepocket_from_pretrain.json"
VOCAB = DATA / "union_vocab" / "base-full__crossdocked__tastepocket_trainonly" / "rgroup_vocab.pkl.gz"
SEED = 911012

# shape kwargs shared by TRAINING and SCORING -- one list, used twice
W512 = ["nnet_module_kwargs.hidden_dim=512"]
FLAV = ["data_module_kwargs.condvec_dim=24", "nnet_module_kwargs.condvec_dim=24"]
POCK = ["data_module_kwargs.condvec_dim=1304", "nnet_module_kwargs.condvec_dim=1304",
        "nnet_module_kwargs.pocket_input_dim=1280", "nnet_module_kwargs.pocket_dim=32",
        "nnet_module_kwargs.pocket_dropout=0.0"]
CLEAN = ["data_module_kwargs.split_by=molecule",
         f"data_module_kwargs.exclude_ids_path={EXCLUDE}"]


@dataclass
class Run:
    name: str
    corpus: str
    epochs: int
    lr: float
    model: List[str]
    extra: List[str] = field(default_factory=list)
    init_from: Optional[str] = None
    batch_size: Optional[int] = None


def runs(arm: str) -> List[Run]:
    tag = "z" if arm == "zinc" else "f"
    s1_corpus = MERGED if arm == "zinc" else FLAVOUR
    s1_epochs = 3 if arm == "zinc" else 30      # ~equal optimiser steps
    s1 = Run(f"chain-{tag}-s1", s1_corpus, s1_epochs, 1e-3,
             W512 + FLAV, CLEAN + ["assembly.enabled=true"])
    s2 = Run(f"chain-{tag}-s2", FLAVOUR, 20, 1e-4,
             W512 + FLAV, CLEAN + ["assembly.enabled=true"],
             init_from=f"{CKPT}/chain-{tag}-s1_best.pt")
    return [s1, s2]


def sh(cmd: List[str], log: Path, gpu: str) -> int:
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "WANDB_MODE": "disabled"})
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        return subprocess.call(cmd, cwd=SRC, stdout=fh, stderr=subprocess.STDOUT, env=env)


def train(r: Run, gpu: str, dry: bool) -> int:
    log = LOGS / f"{r.name}.log"
    if log.exists():
        print(f"  SKIP  {r.name} (log exists)"); return 0
    cmd = [sys.executable, "-u", "run.py",
           f"experiment_name={r.name}",
           f"hydra.run.dir={REPO}/outputs/{r.name}",
           f"random_seed={SEED}",
           f"data_module_kwargs.dataset_version={r.corpus}",
           f"lightning_module_kwargs.learning_rate={r.lr}",
           f"++trainer_kwargs.max_epochs={r.epochs}",
           f"rgroup_library.vocab_path={VOCAB}",
           "rgroup_library.enable_after_epoch=0",
           "wandb.project=null"] + r.model + r.extra
    if r.init_from:
        cmd.append(f"init_weights_from={r.init_from}")
    if r.batch_size:
        cmd.append(f"data_module_kwargs.batch_size={r.batch_size}")
    if dry:
        print("   ", " ".join(cmd[2:])); return 0
    t = time.time()
    rc = sh(cmd, log, gpu)
    print(f"  {r.name:<18} rc={rc}  {(time.time()-t)/60:.1f} min", flush=True)
    return rc


RETR = re.compile(r"RGroupLibraryRetrieval/test\]")


def score(name: str, ckpt: Path, corpus: str, model: List[str], gpu: str,
          extra: Optional[List[str]] = None) -> Dict:
    log = LOGS / f"eval-{name}.log"
    if not log.exists():
        cmd = [sys.executable, "-u", "run.py", "run_mode=test",
               f"experiment_name=eval-{name}",
               f"hydra.run.dir=/tmp/hyd-eval-{name}",
               f"random_seed={SEED}",
               f"+checkpoint_path={ckpt}",
               f"data_module_kwargs.dataset_version={corpus}",
               f"rgroup_library.vocab_path={VOCAB}",
               "rgroup_library.enable_after_epoch=0",
               "wandb.project=null"] + model + (extra or []) + CLEAN
        sh(cmd, log, gpu)
    if not log.exists():
        return {}
    lines = [l for l in log.read_text(errors="ignore").splitlines() if RETR.search(l)]
    if not lines:
        return {"error": "no retrieval line"}
    s = lines[-1]
    g = lambda k: (float(m.group(1)) if (m := re.search(rf"{k}=([0-9.]+)", s)) else None)
    return {"H@1": g("H@1"), "H@10": g("H@10"), "MRR": g("MRR"),
            "novel@10": g("novel_hit@10"), "N": g("N")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--arms", default="zinc,flavour")
    a = ap.parse_args()

    if not a.dry:
        assert VOCAB.exists(), f"train-only vocab missing: {VOCAB}"
        assert EXCLUDE.exists(), f"exclusion list missing: {EXCLUDE}"

    rows = []
    for arm in a.arms.split(","):
        print(f"\n=== ARM {arm.upper()}")
        for r in runs(arm):
            if train(r, a.gpu, a.dry) != 0 and not a.dry:
                print(f"  {r.name} FAILED -- stopping this arm"); break
            if a.dry:
                continue
            ck = CKPT / f"{r.name}_best.pt"
            if not ck.exists():
                print(f"  {r.name}: NO CHECKPOINT"); break
            res = score(r.name, ck, FLAVOUR, W512 + FLAV, a.gpu)
            print(f"    {r.name}  H@1={res.get('H@1')}  H@10={res.get('H@10')}  "
                  f"novel@10={res.get('novel@10')}  N={res.get('N')}", flush=True)
            rows.append({"arm": arm, "run": r.name, "corpus": r.corpus,
                         "epochs": r.epochs, "lr": r.lr, **res})

    if not a.dry and rows:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nresults -> {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
