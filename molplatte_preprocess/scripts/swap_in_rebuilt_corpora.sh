#!/usr/bin/env bash
# Swap the condvec_version 3 rebuilds into place.
#
# Verifies FIRST, swaps second, and never deletes: the corpus being replaced is
# renamed to <name>.v2 and left on disk for the caller to remove once they are
# satisfied. A corpus takes hours of CPU to rebuild and is the input to every
# result in the project.
#
# The checks are the ones that would otherwise fail silently:
#
#   condvec_version   the whole point of the rebuild; a v2 dir swapped in would
#                     look completely normal
#   n_records         a partial build produces a smaller corpus that still loads
#   rgroup_vocab      the build script runs preprocess_flavor.py only, so a
#                     fresh corpus has NO vocabulary. Retrieval would disable
#                     itself with a warning and every run would report no Hit@K
#   rgroup_counts     drives the dataset's common-R-group filter, which falls
#                     back rather than failing when the file is missing
#   hash set          the decomposition did not change, so the R-group hashes
#                     must be IDENTICAL to the corpus being replaced. If they
#                     are not, something other than the condvec moved and the
#                     union library no longer matches the corpus.
set -euo pipefail

D="${D:-/home/mogan/preprocessed/molplatte}"
SRC="${SRC:-/home/mogan/github/MolPLAtte/molplatte_preprocess/src}"
CORPORA="${CORPORA:-flavordb-only coconut-flavordb-filtered coconut-only coconut-flavordb-full}"
METHOD=naveja_recap
export PYTHONPATH="$SRC"

fail=0
echo "== verifying"
for c in $CORPORA; do
  new="$D/$c.new"; old="$D/$c/$METHOD"
  for f in __meta__.json rgroup_vocab.pkl.gz rgroup_counts.json.gz; do
    [ -f "$new/$f" ] || { echo "  FAIL $c: missing $f"; fail=1; }
  done
  [ -d "$old" ] || { echo "  FAIL $c: no existing corpus at $old"; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "== aborted, nothing moved"; exit 1; }

python3 - "$D" "$METHOD" $CORPORA <<'PY' || exit 1
import gzip, json, pickle, sys
from pathlib import Path

D, METHOD, *corpora = sys.argv[1:]
D = Path(D)
bad = False


def hashes(p):
    with gzip.open(p, "rb") as fh:
        v = pickle.load(fh)
    e = v["entries"] if isinstance(v, dict) else v.entries
    return set(e)


for c in corpora:
    new, old = D / f"{c}.new", D / c / METHOD
    mnew = json.loads((new / "__meta__.json").read_text())
    mold = json.loads((old / "__meta__.json").read_text())

    problems = []
    if mnew.get("condvec_version") != 3:
        problems.append(f"condvec_version {mnew.get('condvec_version')} != 3")
    if mnew["n_records"] != mold["n_records"]:
        problems.append(f"n_records {mnew['n_records']:,} != {mold['n_records']:,}")
    hn, ho = hashes(new / "rgroup_vocab.pkl.gz"), hashes(old / "rgroup_vocab.pkl.gz")
    if hn != ho:
        problems.append(f"R-group hash set differs: +{len(hn-ho)} -{len(ho-hn)}")

    tag = "  ok  " if not problems else "  FAIL"
    print(f"{tag} {c:28s} n={mnew['n_records']:>8,}  v{mnew.get('condvec_version')}  "
          f"vocab {len(hn):,}")
    for p in problems:
        print(f"        {p}")
        bad = True

sys.exit(1 if bad else 0)
PY

echo
echo "== swapping (originals kept as <name>.v2)"
for c in $CORPORA; do
  [ -e "$D/$c.v2" ] && { echo "  FAIL $c: $D/$c.v2 already exists, refusing to overwrite"; exit 1; }
  mv "$D/$c" "$D/$c.v2"
  mkdir -p "$D/$c/$METHOD"
  mv "$D/$c.new"/* "$D/$c/$METHOD/"
  rmdir "$D/$c.new"
  echo "  swapped $c   (previous corpus at $c.v2)"
done

echo
echo "== verifying the swapped corpora load"
python3 - "$D" "$METHOD" $CORPORA <<'PY'
import json, sys
from pathlib import Path
D, METHOD, *corpora = sys.argv[1:]
for c in corpora:
    p = Path(D) / c / METHOD
    m = json.loads((p / "__meta__.json").read_text())
    files = [f for f in ("rgroup_vocab.pkl.gz", "rgroup_counts.json.gz")
             if not (p / f).is_file()]
    print(f"  {c:28s} n={m['n_records']:>8,}  condvec_v{m['condvec_version']}  "
          + ("all artifacts present" if not files else f"MISSING {files}"))
PY

echo
echo "== done. Previous corpora are at <name>.v2 and were NOT deleted."
echo "   Remove them with:  rm -rf $D/*.v2"
