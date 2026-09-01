#!/usr/bin/env bash
# Build naveja_recap corpora at several core ratios, plus their R-group libraries.
#
# Motivation: at ratio 0.5 the R-group vocabulary is severely long-tailed and its
# most frequent entries are chemically trivial (*C, *O, *CO). Naveja's `ratio` is
# the core-size threshold -- a RECAP child qualifies as a core only if it holds
# at least that fraction of the molecule's atoms -- so a LOWER ratio admits
# smaller cores, which leaves larger and more varied R-groups.
#
# Measured on flavour chemistry beforehand: k>=2 goes 3.70% (2/3) -> 5.54% (1/2)
# -> 8.34% (1/3), so the effect on decomposition shape is real but modest; the
# question this sweep answers is what it does to the VOCABULARY.
set -uo pipefail
CORPORA="${CORPORA:-/home/mogan/preprocessed/molpallete}"
WORKERS="${WORKERS:-92}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ -f "$REPO/preprocess_flavor.py" ] || { echo "ERROR: wrong repo: $REPO" >&2; exit 1; }

for SPEC in "0.4:r040" "0.3333:r033"; do
  RATIO="${SPEC%%:*}"; TAG="${SPEC##*:}"
  OUT="$CORPORA/coconut-flavordb_full_${TAG}/naveja_recap"
  if [ -f "$OUT/__manifest__.json" ]; then
    echo "== SKIP $TAG (already built)"
  else
    echo "== BUILD ratio=$RATIO -> $TAG"
    python preprocess_flavor.py --source flavordb coconut --method naveja_recap \
      --core-ratio "$RATIO" --output-path "$OUT" \
      --workers "$WORKERS" --layout hash3 --progress-every 100000
  fi
  if [ -f "$OUT/rgroup_vocab.pkl.gz" ]; then
    echo "== SKIP vocab $TAG"
  else
    echo "== VOCAB $TAG"
    python enumerate_rgroups.py --corpus "$OUT" --workers "$WORKERS" --progress-every 100000
  fi
done
echo "== RATIO SWEEP COMPLETE"
