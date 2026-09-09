#!/usr/bin/env bash
# MolPLAtte — one entry point for the three-step pipeline.
#
#   ./run_experiment.sh <step> [options]
#
# Steps:
#   step1     pretrain on wide chemistry (ZINC)         D and E off
#   step2     conditioned training on flavour compounds  D on, E off
#   step3     pocket-aware finetuning (tastepocket)      D and E on
#   assembly  measure how reliable the assembly head is
#   status    what has been built, what has been trained, what is running
#
# Each step PRINTS what it is about to do and why, checks its prerequisites,
# and refuses with a specific message rather than failing halfway through a
# multi-hour run. Pass --dry-run to see the command without running it.
#
# Every step is resumable: nothing is deleted, and a step whose checkpoint
# already exists says so instead of silently retraining over it.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$REPO/molplatte/src"
PREP="$REPO/molplatte_preprocess"
DATA="${DATA:-$HOME/preprocessed/molplatte}"
CK="${CK:-$HOME/checkpoints/molplatte}"
LOG="$CK/hplogs"
GPU="${GPU:-1}"
SEED="${SEED:-911012}"
WANDB_PROJECT="${WANDB_PROJECT:-molplatte}"
UNION="$DATA/union_vocab/zincbase__flavor__crossdocked__tastepocket/rgroup_vocab.pkl.gz"

DRY=0
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; done

c_bold=$'\033[1m'; c_dim=$'\033[2m'; c_red=$'\033[31m'
c_grn=$'\033[32m'; c_yel=$'\033[33m'; c_off=$'\033[0m'

say()  { printf "%s\n" "$*"; }
head_() { printf "\n%s%s%s\n" "$c_bold" "$*" "$c_off"; }
note() { printf "  %s%s%s\n" "$c_dim" "$*" "$c_off"; }
ok()   { printf "  %s✓%s %s\n" "$c_grn" "$c_off" "$*"; }
warn() { printf "  %s!%s %s\n" "$c_yel" "$c_off" "$*"; }
die()  { printf "  %s✗ %s%s\n" "$c_red" "$*" "$c_off" >&2; exit 1; }

need_corpus() {
  local name=$1
  [ -f "$DATA/$name/naveja_recap/__meta__.json" ] \
    || die "corpus '$name' not built. See $PREP/README.md, or run: $0 status"
  [ -f "$DATA/$name/naveja_recap/rgroup_vocab.pkl.gz" ] \
    || die "corpus '$name' has no R-group vocabulary. Run:
       cd $PREP && PYTHONPATH=\$PWD/src python3 enumerate_rgroups.py \\
         --corpus $DATA/$name/naveja_recap --workers 96
     Without it retrieval disables itself and every run reports no Hit@K."
}

need_ckpt() {
  [ -f "$1" ] || die "missing checkpoint: $1
     Run the previous step first."
}

gpu_free() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
  [ -z "$used" ] && { warn "cannot read GPU $GPU"; return 0; }
  if [ "$used" -gt 20000 ]; then
    warn "GPU $GPU already holds ${used} MiB — another job may be running."
    warn "Set GPU=<n> to use a different card."
  fi
}

launch() {
  local exp=$1; shift
  if [ -f "$CK/${exp}_best.pt" ]; then
    warn "checkpoint already exists: $CK/${exp}_best.pt"
    warn "delete it or pass a different experiment name to retrain."
    return 0
  fi
  mkdir -p "$LOG"
  if [ "$DRY" = "1" ]; then
    head_ "would run:"; printf '  %s\n' "$*"; return 0
  fi
  cd "$SRC" || die "no $SRC"
  note "log:   $LOG/$exp.log"
  note "wandb: $WANDB_PROJECT / $exp"
  CUDA_VISIBLE_DEVICES="$GPU" setsid nohup "$@" > "$LOG/$exp.log" 2>&1 < /dev/null &
  local pid=$!
  sleep 25
  if ! kill -0 "$pid" 2>/dev/null && ! grep -q "Training Samples" "$LOG/$exp.log" 2>/dev/null; then
    tail -15 "$LOG/$exp.log"
    die "the run died within 25s — see the log above"
  fi
  ok "started (pid $pid). Follow with:  tail -f $LOG/$exp.log"
}

case "${1:-help}" in

# ---------------------------------------------------------------- step 1
step1)
  head_ "STEP 1 — wide-chemistry pretraining"
  say "  Corpus  : zinc-10m (9.9M molecules, flat across all 89 MW/logP tranches)"
  say "  Trains  : encoder (A), projectors (B), assembly head (C)"
  say "  Condvec : OFF. ZINC has no flavour labels and no pockets, so D and E"
  say "            are zero here -- there is nothing honest to put in them."
  say "  Why     : the assembly head is chemistry-general and needs no labels,"
  say "            so this is the only stage that can train it at scale."
  echo
  need_corpus zinc-10m
  [ -f "$UNION" ] || die "union library missing: $UNION"
  ok "corpus and union library present"
  gpu_free
  launch "pretrain_zinc-10m" \
    python3 -u run.py experiment_name=pretrain_zinc-10m \
      hydra.run.dir="$REPO/molplatte/outputs/pretrain_zinc-10m" \
      random_seed="$SEED" \
      data_module_kwargs.dataset_version=zinc-10m \
      data_module_kwargs.condvec_dim=0 nnet_module_kwargs.condvec_dim=0 \
      assembly.enabled=true \
      rgroup_library.vocab_path="$UNION" \
      ++trainer_kwargs.max_epochs="${EPOCHS:-6}" \
      ++trainer_kwargs.val_check_interval=0.5 \
      wandb.project="$WANDB_PROJECT" wandb.group=step1-pretrain \
      wandb.job_type=pretrain wandb.name=pretrain_zinc-10m
  ;;

# ---------------------------------------------------------------- step 2
step2)
  head_ "STEP 2 — flavour-conditioned training"
  say "  Corpus  : ${CORPUS:-coconut-flavordb-full}"
  say "  Init    : STEP 1 weights (query projector widened 300 -> 324)"
  say "  Condvec : 24 flavour bits ON, pocket OFF"
  say "  Freeze  : ${FREEZE:-nothing}   (FREEZE=encoder to hold ZINC chemistry fixed)"
  echo
  note "Only 3.1% of this corpus carries a real flavour descriptor; the rest is"
  note "'odorless' or 'unknown'. Judge the condvec by a shuffle test, not by loss."
  echo
  need_corpus "${CORPUS:-coconut-flavordb-full}"
  need_ckpt "$CK/pretrain_zinc-10m_best.pt"
  EXPANDED="$CK/pretrain_zinc-10m_expanded24.pt"
  if [ ! -f "$EXPANDED" ]; then
    note "widening the STEP 1 query projector for the 24 flavour bits ..."
    (cd "$SRC" && python3 scripts/expand_checkpoint_for_pocket.py \
       "$CK/pretrain_zinc-10m_best.pt" --out "$EXPANDED" \
       --flavor-dim 0 --pocket-dim 24) || die "checkpoint expansion failed"
  fi
  ok "warm start ready: $(basename "$EXPANDED")"
  gpu_free
  FREEZE_ARG=""
  [ -n "${FREEZE:-}" ] && FREEZE_ARG="freeze=[$FREEZE]"
  launch "contrain_${CORPUS:-coconut-flavordb-full}" \
    python3 -u run.py experiment_name="contrain_${CORPUS:-coconut-flavordb-full}" \
      hydra.run.dir="$REPO/molplatte/outputs/contrain_${CORPUS:-coconut-flavordb-full}" \
      random_seed="$SEED" init_weights_from="$EXPANDED" \
      data_module_kwargs.dataset_version="${CORPUS:-coconut-flavordb-full}" \
      data_module_kwargs.condvec_dim=24 nnet_module_kwargs.condvec_dim=24 \
      assembly.enabled=true ${FREEZE_ARG} \
      rgroup_library.vocab_path="$UNION" \
      ++trainer_kwargs.max_epochs="${EPOCHS:-20}" \
      wandb.project="$WANDB_PROJECT" wandb.group=step2-flavor-contrain \
      wandb.job_type=contrain wandb.name="contrain_${CORPUS:-coconut-flavordb-full}"
  ;;

# ---------------------------------------------------------------- step 3
step3)
  head_ "STEP 3 — pocket-aware finetuning"
  say "  Corpus  : tastepocket_corpus (243 records, 5-fold CV then a full-data model)"
  say "  Init    : STEP 2 weights (query projector widened 324 -> 356)"
  say "  Condvec : 24 flavour bits + 1280-d ESM-2 pocket"
  echo
  warn "269 records is very few. The 2026-09-05 result was that neither the"
  warn "pocket half nor this finetuning stage changed retrieval measurably"
  warn "(+0.0008 and -0.0106 H@1). See molplatte/docs/step2_pocket_results_*.md"
  echo
  need_corpus tastepocket_corpus
  need_ckpt "$CK/contrain_${CORPUS:-coconut-flavordb-full}_best.pt"
  [ -f "$DATA/tastepocket_corpus/naveja_recap/folds.json" ] \
    || die "no folds.json — build it with build_tastepocket_dataset.py --folds-out"
  ok "corpus, folds and STEP 2 checkpoint present"
  gpu_free
  if [ "$DRY" = "1" ]; then
    head_ "would run:"; say "  bash $SRC/scripts/run_step2_pocket_cv.sh"; exit 0
  fi
  STEP1="$CK/contrain_${CORPUS:-coconut-flavordb-full}_best.pt" \
    bash "$SRC/scripts/run_step2_pocket_cv.sh"
  python3 "$SRC/scripts/summarise_step2_cv.py"
  ;;

# ---------------------------------------------------------------- assembly
assembly)
  head_ "Assembly-head reliability"
  CKPT="${CKPT:-$CK/pretrain_zinc-10m_best.pt}"
  need_ckpt "$CKPT"
  cd "$SRC" || exit 1
  CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/evaluate_assembly.py \
    --checkpoint "$CKPT" --vocab "$UNION" \
    --corpus "$DATA/${CORPUS:-coconut-flavordb-full}/naveja_recap" \
    --n "${N:-300}" --condvec-dim "${CONDVEC_DIM:-0}"
  ;;

# ---------------------------------------------------------------- status
status)
  head_ "Corpora"
  printf "  %-28s %10s %10s %s\n" NAME RECORDS VOCAB CONDVEC
  for d in "$DATA"/*/naveja_recap/__meta__.json; do
    [ -f "$d" ] || continue
    python3 - "$d" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); m = json.loads(p.read_text())
v = p.parent / "rgroup_vocab.pkl.gz"
print(f"  {p.parent.parent.name:<28} {m['n_records']:>10,} "
      f"{'yes' if v.is_file() else 'MISSING':>10} "
      f"v{m.get('condvec_version')} {m.get('condvec_mode')}")
PY
  done

  head_ "Checkpoints"
  ls -1t "$CK"/*_best.pt 2>/dev/null | head -8 | while read -r f; do
    printf "  %-42s %6s MB  %s\n" "$(basename "$f")" \
      "$(( $(stat -c%s "$f") / 1000000 ))" "$(stat -c %y "$f" | cut -c1-16)"
  done || note "none yet"

  head_ "Running"
  n=$(ps -eo pid,cmd | awk '$2=="python3" && /run\.py/ && /experiment_name/' | wc -l)
  if [ "$n" -gt 0 ]; then
    ps -eo pid,etime,cmd | awk '$3=="python3" && /experiment_name/' \
      | sed 's/.*experiment_name=\([^ ]*\).*/  \1/' | sort -u | while read -r e; do
        printf "  %s\n" "$e"
      done
  else
    note "nothing training"
  fi
  ;;

*)
  head_ "MolPLAtte experiment runner"
  say "  $0 step1      wide-chemistry pretraining (ZINC)"
  say "  $0 step2      flavour-conditioned training"
  say "  $0 step3      pocket-aware finetuning"
  say "  $0 assembly   measure assembly-head reliability"
  say "  $0 status     corpora, checkpoints, running jobs"
  echo
  say "  Options are environment variables:"
  say "    GPU=1              which card                 (default 1)"
  say "    EPOCHS=20          override the step default"
  say "    CORPUS=name        override the step corpus"
  say "    FREEZE=encoder     freeze components: encoder, projectors, assembly, pocket"
  say "    SEED=911012        random seed"
  say "    --dry-run          print the command without running it"
  echo
  say "  Example:  FREEZE=encoder EPOCHS=10 $0 step2"
  ;;
esac
