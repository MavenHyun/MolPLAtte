#!/usr/bin/env python3
"""Drop pockets whose SENSOR protein comes from an excluded clade.

Filters on ``organism`` (resolved from the curated ``sensor_organism``), NOT on
the family label. The labels are not reliable for this: 34 pockets from 16 PDB
entries are bovine/porcine/canine lipocalins filed under "OBP - insect odorant/
pheromone binding protein", and a family-name filter would delete those genuine
vertebrate odorant carriers along with the insect ones.

Writes the kept records and prints exactly what was removed -- a silent filter
that drops 15% of a dataset is how a corpus quietly stops meaning what its
documentation says.
"""
from __future__ import annotations
import argparse, collections, json, sys
from pathlib import Path

INSECT_GENERA = {
    "Acyrthosiphon", "Adelphocoris", "Aedes", "Agrotis", "Amyelois", "Anopheles",
    "Anoplophora", "Antheraea", "Apis", "Athetis", "Bombus", "Bombyx", "Camponotus",
    "Ceratitis", "Chilo", "Culex", "Cydia", "Dendrolimus", "Drosophila", "Eogystia",
    "Epiphyas", "Helicoverpa", "Holotrichia", "Leptinotarsa", "Locusta", "Lygus",
    "Machilis", "Mamestra", "Manduca", "Monochamus", "Nasonia", "Ostrinia",
    "Phormia", "Plutella", "Rhyparobia", "Sesamia", "Sitophilus", "Spodoptera",
    "Tribolium", "Varroa",
}
# arthropods that are not insects but belong to the same exclusion intent
ARTHROPOD_GENERA = {"Varroa", "Tetranychus", "Ixodes"}

CLADES = {"insect": INSECT_GENERA | ARTHROPOD_GENERA}


def genus(org: str) -> str:
    parts = (org or "").split()
    return parts[0] if parts else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pockets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude", default="insect", choices=sorted(CLADES))
    a = ap.parse_args()

    drop_genera = CLADES[a.exclude]
    rows = [json.loads(l) for l in open(a.pockets)]
    keep, dropped = [], []
    for r in rows:
        (dropped if genus(r.get("organism", "")) in drop_genera else keep).append(r)

    unknown = [r for r in keep if not (r.get("organism") or "").strip()]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for r in keep:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    pdb = lambda rs: len({r["pdb_id"] for r in rs})
    rcp = lambda rs: len({str(r.get("uniprot")) for r in rs})
    print(f"  in   {len(rows):>5} pockets  {pdb(rows):>4} PDB  {rcp(rows):>4} receptors")
    print(f"  DROP {len(dropped):>5} pockets  {pdb(dropped):>4} PDB  {rcp(dropped):>4} receptors "
          f"(clade={a.exclude})")
    print(f"  keep {len(keep):>5} pockets  {pdb(keep):>4} PDB  {rcp(keep):>4} receptors")
    if unknown:
        print(f"  NOTE {len(unknown)} kept records have no organism and were NOT filtered")
    print("\n  dropped by organism:")
    for o, n in collections.Counter(r.get("organism", "?") for r in dropped).most_common():
        print(f"    {n:>5}  {o}")
    print("\n  dropped by family:")
    for f, n in collections.Counter((r.get("families") or ["?"])[0] for r in dropped).most_common():
        print(f"    {n:>5}  {f[:58]}")
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
