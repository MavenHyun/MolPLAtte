#!/usr/bin/env bash
# Build every MolPallete pretraining corpus.
#
# macfrag is the default recipe: naveja_recap -- MolPLA's own method -- collapses
# to ~1 R-group per decomposition on flavor chemistry, which makes MolPLA's
# islinked subset enumeration degenerate.  naveja and synton are built anyway for
# the comparison the brief asks for.  See README.md for the measured table.
set -euo pipefail

CORPORA="${CORPORA:-/home/mogan/corpora/molpallete}"
WORKERS="${WORKERS:-88}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

build () {
  local source=$1 method=$2 version=$3; shift 3
  local out="$CORPORA/$version/$method"
  if [ -f "$out/__meta__.json" ]; then
    echo "== SKIP $version/$method (already built)"; return 0
  fi
  echo "== BUILD $version/$method -> $out"
  python preprocess_flavor.py \
    --source "$source" --method "$method" --output-path "$out" \
    --workers "$WORKERS" --batch-size 100 --layout hash3 \
    --progress-every 25000 "$@"
}

# FlavorDB -- 25,595 compounds, the in-domain set.
build flavordb macfrag      flavor_v1
build flavordb naveja_recap flavor_v1
build flavordb bemis_murcko flavor_v1
build flavordb synton       flavor_v1

# COCONUT -- 737,343 records collapsing to 489,395 distinct compounds.
build coconut macfrag      coconut_v1
build coconut naveja_recap coconut_v1
build coconut bemis_murcko coconut_v1
build coconut synton       coconut_v1

echo "== ALL CORPORA BUILT"
du -sh "$CORPORA"/*/* 2>/dev/null || true
