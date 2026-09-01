#!/usr/bin/env bash
# Rebuild the four named corpus variants at build tag r333-m2-h4-flavor24.
#
# Settings are copied from the __meta__.json of the corpora they replace, so the
# ONLY intended difference is the flavor-condvec fix (24b75d1): before it, the
# condvec collapsed to a single MW>350 bit in every corpus.
#
# Builds into <name>.new and verifies flavor content before anything is swapped.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPORA="${CORPORA:-/home/mogan/preprocessed/molplatte}"
ANN="$CORPORA/annotations"
WORKERS="${WORKERS:-96}"
cd "$REPO"
export PYTHONPATH="$REPO/src"

common=(
  --method naveja_recap
  --core-ratio 0.3333333333333333
  --max-cores 4 --max-rgroups 8 --min-rgroup-atoms 2
  --keep-stereo --no-neutralise
  --min-heavy-atoms 5 --max-heavy-atoms 50
  --condvec-mode flavor
  --flavor-measured "$ANN/flavor_measured.jsonl"
  --flavor-mined    "$ANN/flavor_documented.jsonl"
  --layout hash3
  --workers "$WORKERS" --batch-size 100 --progress-every 25000
)

build () {
  local name=$1; shift
  local out="$CORPORA/$name.new"
  echo "=========================================================="
  echo "== BUILD $name  ($(date +%H:%M:%S))"
  rm -rf "$out"
  python preprocess_flavor.py --output-path "$out" "${common[@]}" "$@"
  echo "== DONE $name  ($(date +%H:%M:%S))"
}

# smallest first, so a systematic problem surfaces in minutes not hours
build flavordb-only              --source flavordb
build coconut-flavordb-filtered  --source flavordb coconut --include-ids "$ANN/flavor_subset_ids.txt"
build coconut-only               --source coconut
build coconut-flavordb-full      --source flavordb coconut

echo "== ALL FOUR BUILT"
