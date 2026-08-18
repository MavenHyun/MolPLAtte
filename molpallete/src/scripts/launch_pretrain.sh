#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# MolPallete pretraining launcher.
#
# MolPLA's three contrastive objectives over the anchored G/P/R/Q views, on the
# flavor + natural-product corpus. There is no MolDAM-style baseline to beat
# yet -- this is the first run of the repo, so the numbers it produces ARE the
# baseline. Record them in docs/ before changing anything.
#
# Selection metric comes from $CONFIG (`checkpoint:` / `early_stopping:`
# blocks); this script does not impose one. The headline retrieval number to
# watch is `val/faiss/r@10` on the DEDUPLICATED gallery -- and it must be read
# against the R-group frequency prior, not against random. A skewed vocabulary
# makes a constant predictor look strong; MolDAM lost seven checkpoints to
# that mistake before measuring the prior.
# ---------------------------------------------------------------------------
set -euo pipefail

EXP="${EXP:-molpallete_flavor_v1_macfrag_first_run}"
REPO="${REPO:-/home/mogan/github/MolPallete/molpallete}"
LOGDIR="${LOGDIR:-/home/mogan/logs/molpallete}"
CONFIG="${CONFIG:-config}"
OUTDIR="$REPO/outputs/$EXP"
mkdir -p "$LOGDIR"

# ----------------------------- wandb ---------------------------------------
# Leave WANDB_PROJECT empty to disable wandb entirely (the run then falls back
# to Lightning's default local logger -- training is unaffected).
# Do NOT hardcode the key here -- this file is tracked in git. Run `wandb login`
# once (writes ~/.netrc), or `export WANDB_API_KEY=...` in your shell first.
# Only export it onward if it is actually non-empty: an exported blank key makes
# wandb fail with "No API key configured" instead of falling back cleanly.
[[ -n "${WANDB_API_KEY:-}" ]] && export WANDB_API_KEY
WANDB_PROJECT="${WANDB_PROJECT:-NoahsFarm_MolPallete}"
WANDB_ENTITY="${WANDB_ENTITY:-}"          # optional; or set via `wandb login`
WANDB_RUN_NAME="$EXP"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"

WANDB_ARGS=()
if [[ -n "$WANDB_PROJECT" ]]; then
  WANDB_ARGS=(
    "wandb.project=$WANDB_PROJECT"
    "wandb.name=$WANDB_RUN_NAME"
    "wandb.log_model=$WANDB_LOG_MODEL"
  )
  [[ -n "$WANDB_ENTITY" ]] && export WANDB_ENTITY
fi
# ---------------------------------------------------------------------------

# Fail fast rather than 100 epochs into nothing: if wandb is requested but no
# credential exists, stop here instead of crashing inside Trainer.fit.
if [[ -n "$WANDB_PROJECT" && -z "${WANDB_API_KEY:-}" ]] \
   && ! grep -qs "api.wandb.ai" ~/.netrc; then
  echo "ERROR: WANDB_PROJECT is set but no wandb credential found." >&2
  echo "  Run 'wandb login' once, or 'export WANDB_API_KEY=...' before this script." >&2
  echo "  (Or clear WANDB_PROJECT to train without wandb.)" >&2
  exit 1
fi

# Refuse to clobber a previous run's outputs. Hydra would happily write into an
# existing directory and interleave two runs' prediction tables, which is
# unrecoverable after the fact.
if [ -e "$OUTDIR" ] && [ -n "$(ls -A "$OUTDIR" 2>/dev/null)" ]; then
  echo "ERROR: $OUTDIR already exists and is non-empty. Refusing to overwrite." >&2
  echo "  Set EXP=<new name>, or remove it explicitly if you intend to rerun." >&2
  exit 1
fi
if [ -e "$LOGDIR/$EXP.log" ]; then
  echo "ERROR: $LOGDIR/$EXP.log already exists. Refusing to overwrite." >&2
  echo "  Set EXP=<new name>, or move the old log aside." >&2
  exit 1
fi

cd "$REPO/src"

nohup setsid python -u run.py --config-name "$CONFIG" \
  experiment_name="$EXP" \
  hydra.run.dir="../outputs/$EXP" \
  "${WANDB_ARGS[@]}" \
  > "$LOGDIR/$EXP.log" 2>&1 < /dev/null &

sleep 3
# $! is the nohup wrapper, which setsid replaces -- resolve the real PID
pgrep -f "run.py.*experiment_name=$EXP" | head -1 > "$LOGDIR/$EXP.pid"
echo "launched $EXP  pid=$(cat "$LOGDIR/$EXP.pid")"
echo "  log     : $LOGDIR/$EXP.log"
echo "  outputs : $OUTDIR"
echo "  ckpt    : /home/mogan/checkpoints/${EXP}_best.pt"
echo "  tables  : $OUTDIR/prediction_tables/"
