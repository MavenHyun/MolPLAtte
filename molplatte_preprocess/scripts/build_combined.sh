#!/usr/bin/env bash
# Build the primary MolPLAtte pretraining corpus: FlavorDB + COCONUT combined.
#
# This is the corpus the pretraining runs use. The per-source corpora under
# flavordb_full/ and coconut_full/ remain as the source ablation and the decomposition
# method comparison; note they predate always-on deduplication.
set -euo pipefail

CORPORA="${CORPORA:-/home/mogan/preprocessed/molplatte}"
WORKERS="${WORKERS:-88}"
VERSION="${VERSION:-coconut-flavordb_full}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ -f "$REPO/preprocess_flavor.py" ] || { echo "ERROR: wrong repo: $REPO" >&2; exit 1; }

for METHOD in "$@"; do
  OUT="$CORPORA/$VERSION/$METHOD"
  if [ -f "$OUT/__manifest__.json" ]; then
    echo "== SKIP $VERSION/$METHOD (already built)"
  else
    echo "== BUILD $VERSION/$METHOD"
    python preprocess_flavor.py \
      --source flavordb coconut --method "$METHOD" \
      --output-path "$OUT" --workers "$WORKERS" \
      --layout hash3 --progress-every 50000
  fi

  if [ -f "$OUT/rgroup_vocab.pkl.gz" ]; then
    echo "== SKIP vocab $VERSION/$METHOD (already built)"
  else
    echo "== VOCAB $VERSION/$METHOD"
    python enumerate_rgroups.py --corpus "$OUT" --workers "$WORKERS" --progress-every 50000
  fi
done
echo "== COMBINED CORPUS READY"
