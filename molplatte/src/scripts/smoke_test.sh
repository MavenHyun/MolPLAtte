#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# MolPLAtte smoke test -- does the whole stack run end to end?
#
# Runs in the FOREGROUND on purpose: the only useful output is the traceback,
# and a backgrounded smoke test that dies silently is worse than none.
#
# Two stages, cheapest first:
#   1. fast_dev_run  -- 1 train + 1 val batch, no checkpointing, no logger.
#                       Catches shape errors, missing batch keys, device bugs.
#   2. config_debug  -- 2 epochs x 5 steps, num_workers=0, real checkpointing
#                       and callbacks. Catches epoch-boundary and callback bugs
#                       that fast_dev_run skips entirely (it disables both).
#
# Stage 2 does NOT exercise FAISSRetrieval / PredictionTable: both are gated on
# epoch >= 3 in main.get_callbacks, and a 5-step corpus would produce
# meaningless r@k anyway. Override with `enable_after_epoch` if you need to.
# ---------------------------------------------------------------------------
set -euo pipefail

# derive the repo root from this script's own location: .../molplatte/src/scripts/x.sh
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUTDIR="$REPO/outputs/smoke"

# Refuse to clobber: a stale outputs/smoke makes a passing run look like it
# wrote nothing new.
if [ -e "$OUTDIR" ] && [ -n "$(ls -A "$OUTDIR" 2>/dev/null)" ]; then
  echo "ERROR: $OUTDIR already exists and is non-empty. Refusing to overwrite." >&2
  echo "  rm -rf $OUTDIR   to rerun." >&2
  exit 1
fi

cd "$REPO/src"
export HYDRA_FULL_ERROR=1

echo "=== stage 1/2: fast_dev_run ==============================================="
python -u run.py --config-name config_debug \
  experiment_name=smoke \
  hydra.run.dir="../outputs/smoke/fast_dev_run" \
  wandb.project=null \
  ++trainer_kwargs.fast_dev_run=1

echo
echo "=== stage 2/2: config_debug (2 epochs x 5 steps) =========================="
python -u run.py --config-name config_debug \
  experiment_name=smoke \
  hydra.run.dir="../outputs/smoke/debug" \
  wandb.project=null

echo
echo "smoke test PASSED"
echo "  outputs : $OUTDIR"
