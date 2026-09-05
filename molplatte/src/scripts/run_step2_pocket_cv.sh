#!/usr/bin/env bash
# STEP 2: pocket-conditioned finetuning on tastepocket.
#
# Five CV folds to SIZE the effect, then one full-data run to PRODUCE the
# deliverable checkpoint. 269 records is far too few for a single holdout --
# a 15% test set is ~40 records and the seed would move the answer more than
# the model does -- so the folds exist to estimate, not to select.
#
# Each fold holds out whole connected components of the (ligand, receptor)
# bipartite graph, so a held-out receptor's ligands are absent from training
# too. Verified at the dataloader level: all five folds disjoint on molecules,
# ligands AND receptors.
#
# Every run warm-starts from the SAME expanded STEP 1 checkpoint. The expansion
# zero-pads the query projector's new pocket columns, and PocketConditioning
# zero-initialises its own output, so each fold begins numerically identical to
# the pretrained model and has to earn any departure from it.
#
# GPU1 ONLY -- GPU0 runs other jobs.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO/src"
DATA="${DATA:-$HOME/preprocessed/molplatte}"
CKPT="${CKPT:-$HOME/checkpoints/molplatte}"
LOGDIR="$CKPT/hplogs"
GPU="${GPU:-1}"

STEP1="${STEP1:-$CKPT/s1-pretrain-full-s911012_best.pt}"
EXPANDED="${EXPANDED:-$CKPT/s1-pretrain-expanded.pt}"
CORPUS="${CORPUS:-tastepocket_corpus}"
VOCAB="${VOCAB:-$DATA/union_vocab/base-full__crossdocked__tastepocket/rgroup_vocab.pkl.gz}"
EPOCHS="${EPOCHS:-40}"
SEED="${SEED:-911012}"
POCKET_DIM="${POCKET_DIM:-32}"
POCKET_DROPOUT="${POCKET_DROPOUT:-0.1}"
FOLDS="${FOLDS:-0 1 2 3 4}"

# Corpus width: 24 flavour bits + the raw ESM-2 half.
CONDVEC_DIM=1304
POCKET_INPUT_DIM=1280

mkdir -p "$LOGDIR"
cd "$SRC"

if [ ! -f "$STEP1" ]; then
  echo "no STEP 1 checkpoint at $STEP1" >&2
  echo "run STEP 1 first, or set STEP1=/path/to/checkpoint" >&2
  exit 1
fi

echo "== expanding the STEP 1 checkpoint for the pocket half"
python3 scripts/expand_checkpoint_for_pocket.py "$STEP1" \
  --out "$EXPANDED" --pocket-dim "$POCKET_DIM"

common_args () {
  echo "data_module_kwargs.dataset_version=$CORPUS \
        data_module_kwargs.condvec_dim=$CONDVEC_DIM \
        nnet_module_kwargs.condvec_dim=$CONDVEC_DIM \
        nnet_module_kwargs.pocket_input_dim=$POCKET_INPUT_DIM \
        nnet_module_kwargs.pocket_dim=$POCKET_DIM \
        nnet_module_kwargs.pocket_dropout=$POCKET_DROPOUT \
        init_weights_from=$EXPANDED \
        rgroup_library.vocab_path=$VOCAB \
        rgroup_library.enable_after_epoch=0 \
        random_seed=$SEED \
        ++trainer_kwargs.max_epochs=$EPOCHS"
}

# --- folds: estimate ------------------------------------------------------
for k in $FOLDS; do
  exp="s2-pocket-fold${k}-s${SEED}"
  if [ -f "$LOGDIR/$exp.log" ]; then echo "  SKIP $exp (log exists)"; continue; fi
  echo "== fold $k"
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u run.py \
    experiment_name="$exp" \
    hydra.run.dir="$REPO/outputs/$exp" \
    +data_module_kwargs.cv_fold="$k" \
    $(common_args) \
    wandb.project=molplatte \
    wandb.group=step2-pocket-cv \
    wandb.job_type="fold${k}" \
    wandb.name="$exp" \
    > "$LOGDIR/$exp.log" 2>&1
  grep -E "RGroupLibraryRetrieval/val" "$LOGDIR/$exp.log" | tail -1 || true
done

# --- full data: deliver ---------------------------------------------------
# No holdout at all. val_split=0 removes the validation loop, so EarlyStopping
# is switched off (it raises without its metric) and checkpointing monitors
# train/loss (SaveBestModelCheckpoint would otherwise never fire and the run
# would finish having written nothing).
exp="s2-pocket-final-s${SEED}"
if [ -f "$LOGDIR/$exp.log" ]; then
  echo "  SKIP $exp (log exists)"
else
  echo "== final: all 269 records, no holdout"
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u run.py \
    experiment_name="$exp" \
    hydra.run.dir="$REPO/outputs/$exp" \
    data_module_kwargs.val_split=0.0 \
    data_module_kwargs.test_split=0.0 \
    early_stopping.enabled=false \
    +checkpoint.monitor=train/loss \
    +checkpoint.mode=min \
    ++trainer_kwargs.limit_val_batches=0 \
    ++trainer_kwargs.num_sanity_val_steps=0 \
    rgroup_library.enabled=false \
    $(common_args) \
    wandb.project=molplatte \
    wandb.group=step2-pocket-cv \
    wandb.job_type=final \
    wandb.name="$exp" \
    > "$LOGDIR/$exp.log" 2>&1
fi

echo
echo "== STEP 2 COMPLETE"
echo "   folds     $CKPT/s2-pocket-fold*_best.pt"
echo "   deliverable $CKPT/${exp}_best.pt"
echo
echo "   Read novel_hit@K, not just hit@K: a healthy hit@K with a collapsed"
echo "   novel_hit@K means the model is retrieving chemistry it already knew."
echo "   Fold 1 is 52/58 TRPV1+TRPA1, both single-component families, so it"
echo "   measures generalisation to an unseen receptor FAMILY -- a harder"
echo "   question than the other folds ask. Read it separately, not averaged in."
