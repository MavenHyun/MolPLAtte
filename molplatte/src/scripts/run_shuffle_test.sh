#!/usr/bin/env bash
# Condvec shuffle test: does the condition vector leak, and does it help?
#
# Three arms:
#   nocond  no condition vector at all              (capacity baseline)
#   shuf    condvec PERMUTED across molecules       (capacity, no information)
#   cond    real condvec                            (capacity + information)
#
# cond - shuf isolates the INFORMATION in the vector; shuf - nocond is the
# capacity a 24-wide input adds regardless of content. A large cond - shuf on a
# vector derived from the target would indicate leakage; a small one with tight
# seed spread indicates the condition simply does not help.
#
# The previous run of this test (2026-08-31) is void: the condvec then collapsed
# to a bare MW>350 bit, so its "real condition" arm carried no flavour at all.
# It also ran one seed per arm, which cannot separate a 2% effect from the
# +-1.5% seed spread -- hence SEEDS below.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORPUS="${CORPUS:-coconut-flavordb-filtered}"
EPOCHS="${EPOCHS:-25}"
SEEDS="${SEEDS:-911012 20260901 424242}"
GPU="${GPU:-1}"
CONC="${CONC:-3}"            # concurrent runs on one GPU
LOGDIR=/home/mogan/checkpoints/molplatte/hplogs
mkdir -p "$LOGDIR"
cd "$REPO/src"

arm_args () {                # $1 = arm name -> hydra overrides
  case "$1" in
    nocond) echo "data_module_kwargs.condvec_dim=0  nnet_module_kwargs.condvec_dim=0  data_module_kwargs.shuffle_condvec=false" ;;
    cond)   echo "data_module_kwargs.condvec_dim=24 nnet_module_kwargs.condvec_dim=24 data_module_kwargs.shuffle_condvec=false" ;;
    shuf)   echo "data_module_kwargs.condvec_dim=24 nnet_module_kwargs.condvec_dim=24 data_module_kwargs.shuffle_condvec=true"  ;;
  esac
}

launch () {
  local arm=$1 seed=$2
  local exp="sh2-${CORPUS##*-}-${arm}-s${seed}"
  local out="$REPO/outputs/$exp"
  if [ -f "$LOGDIR/$exp.log" ]; then echo "  SKIP $exp (log exists)"; return 0; fi
  rm -rf "$out"
  CUDA_VISIBLE_DEVICES="$GPU" nohup setsid python -u run.py \
    experiment_name="$exp" \
    hydra.run.dir="$out" \
    random_seed="$seed" \
    data_module_kwargs.dataset_version="$CORPUS" \
    $(arm_args "$arm") \
    ++trainer_kwargs.max_epochs="$EPOCHS" \
    wandb.project=null \
    > "$LOGDIR/$exp.log" 2>&1 < /dev/null &
  echo "  launched $exp"
}

# Count RUNS, not processes. Each run spawns ~10 dataloader workers that all
# carry the parent cmdline, so a plain pgrep|wc -l reports 10 for one run and
# trips CONC immediately -- which silently serialised the whole sweep.
running () {
  # Anchor on the python cmdline for two reasons: each run spawns ~10 dataloader
  # workers that all carry the parent argv (so a plain count reports 10 for one
  # run and trips CONC immediately), and a bare pgrep -f also matches any
  # monitoring shell whose own argv contains this pattern.
  pgrep -af "^python -u run\.py .*experiment_name=sh2-" 2>/dev/null \
    | sed 's/.*experiment_name=\([^ ]*\).*/\1/' | sort -u | wc -l
}

for seed in $SEEDS; do
  for arm in nocond shuf cond; do
    while [ "$(running)" -ge "$CONC" ]; do sleep 60; done
    launch "$arm" "$seed"
    sleep 20
  done
done

echo "== all launched; waiting for completion"
while [ "$(running)" -gt 0 ]; do sleep 60; done
echo "== SHUFFLE TEST COMPLETE ($CORPUS, epochs=$EPOCHS)"
