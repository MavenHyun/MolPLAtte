#!/usr/bin/env bash
# Build the R-group library vocabulary for every corpus.
#
# The vocabulary is the static half of MolPLA's R-Group Retrieval library: the
# distinct R-groups, their canonical masked graphs, counts and condition vectors.
# The embedded FAISS index is the model-dependent half and is rebuilt at
# validation time by callbacks/RGroupLibraryRetrieval.py.
set -euo pipefail

CORPORA="${CORPORA:-/home/mogan/preprocessed/molpallete}"
WORKERS="${WORKERS:-88}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Fail loudly if we are not where enumerate_rgroups.py lives. An earlier run of
# this script was launched from a copy that had been written into the training
# repo by mistake; it resolved $REPO there, could not find the driver, and the
# only symptom was a silent "can't open file" buried in a nohup log.
[ -f "$REPO/enumerate_rgroups.py" ] || {
  echo "ERROR: enumerate_rgroups.py not found in $REPO -- wrong repo?" >&2; exit 1; }

for corpus in "$CORPORA"/*/*/; do
  [ -f "$corpus/__meta__.json" ] || continue
  [ -f "$corpus/__manifest__.json" ] || { echo "== SKIP $corpus (still building)"; continue; }
  if [ -f "$corpus/rgroup_vocab.pkl.gz" ]; then
    echo "== SKIP $(basename "$(dirname "$corpus")")/$(basename "$corpus") (vocab exists)"; continue
  fi
  echo "== VOCAB $(basename "$(dirname "$corpus")")/$(basename "$corpus")"
  python enumerate_rgroups.py --corpus "$corpus" --workers "$WORKERS" --progress-every 100000
done
echo "== ALL VOCABULARIES BUILT"
