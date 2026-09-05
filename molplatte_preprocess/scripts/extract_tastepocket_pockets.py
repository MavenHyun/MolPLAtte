#!/usr/bin/env python3
"""Extract the 10 A pocket around every kept cognate ligand in the T1 set.

Emits one record per (pdb_id, ligand instance). Each carries the pocket residue
list AND, for every chain the pocket touches, that chain's full one-letter
sequence plus the 0-based indices of its pocket residues within it.

The indices are what makes a sequence model usable here. A protein language
model embeds a chain end to end; the pocket is then a *geometric* subset of
those per-residue embeddings. So residue selection stays 3D even when the
embedding is 1D -- which matters for GPCRs, where pocket residues are far apart
in sequence (in 8F76 the propionate site draws on 100-108 and 151-157).

10 A matches the ``pocket10`` convention of CrossDocked's processed LMDB, so
pockets from the two sources are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import prody  # noqa: E402

from molplatte_prep.pocket_ligands import _parse_structure, pocket_residues  # noqa: E402

prody.confProDy(verbosity="none")

AA3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}

#: Below this a "pocket" is a surface contact, not an enclosed site. Chosen to
#: be permissive -- the point is to drop degenerate cases, not to curate.
MIN_POCKET_RESIDUES = 8

#: A chain with at least this many amino acids is the protein, not a ligand.
#: Used to tell a free amino-acid LIGAND from the polymer residues that share
#: its name. mmCIF records free amino acids as ATOM rather than HETATM -- in
#: 7DTU the tryptophan agonist is chain G, one residue, 15 atoms (it has OXT),
#: while chains A/B are 778-residue polymers holding 14 backbone tryptophans
#: each. So `hetero` cannot find it and a bare `resname TRP` match sweeps up
#: the whole protein: 30 "ligands" from one structure, 445 across the set.
#: Every pocket built around a backbone residue is meaningless, and nothing
#: about it errors -- the pockets are the right shape and full of real atoms.
MIN_POLYMER_CHAIN = 20


def chain_sequence(structure, chid):
    """(one-letter sequence, {resnum: index}) for one chain's CA trace.

    Keyed on residue number rather than position because deposited structures
    have gaps; using enumerate() directly would silently shift every index
    after the first missing loop.
    """
    sel = structure.select(f"protein and chain {chid} and name CA")
    if sel is None:
        return "", {}
    seq, index_of = [], {}
    for res in sel.getHierView().iterResidues():
        code = AA3to1.get(res.getResname().strip().upper())
        if code is None:
            continue
        index_of[int(res.getResnum())] = len(seq)
        seq.append(code)
    return "".join(seq), index_of


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.home() / "datasets/tastepocket")
    ap.add_argument("--ligands", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cutoff", type=float, default=10.0)
    args = ap.parse_args()

    kept = {}
    for line in args.ligands.open():
        r = json.loads(line)
        if r["ok"]:
            kept[r["ccd"]] = r

    entries = json.loads((args.root / "data" / "taste_odor_pdb.json").read_text())
    t1 = [e for e in entries if e.get("tier") == "T1_ligand_complex"]

    cifdir = args.root / "structures" / "cif"
    stats: Counter = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0

    with args.out.open("w") as fh:
        for e in t1:
            pdb_id = e["id"]
            codes = [lig[0] for lig in (e.get("cognate_ligands") or [])
                     if lig[0] in kept]
            if not codes:
                stats["entry_no_kept_ligand"] += 1
                continue

            cif = cifdir / f"{pdb_id}.cif"
            if not cif.exists():
                stats["cif_missing"] += 1
                continue
            try:
                st = _parse_structure(cif)
            except Exception:  # noqa: BLE001
                stats["cif_unparsable"] += 1
                continue
            if st is None:
                stats["cif_unparsable"] += 1
                continue

            # Chains long enough to be the protein itself. A residue outside
            # them is a ligand even when it is recorded as ATOM.
            polymer_chains = set()
            for chain in st.getHierView().iterChains():
                n_aa = sum(1 for r in chain.iterResidues()
                           if r.getResname().strip().upper() in AA3to1)
                if n_aa >= MIN_POLYMER_CHAIN:
                    polymer_chains.add(chain.getChid())

            seq_cache = {}
            for code in codes:
                named = st.select(f"resname {code}")
                if named is None:
                    stats["ligand_absent_from_coords"] += 1
                    continue
                # Keep only copies that are NOT part of the polymer: either
                # flagged hetero, or sitting in a chain too short to be protein.
                instances = [
                    r for r in named.getHierView().iterResidues()
                    if r.getChid() not in polymer_chains
                ]
                het = st.select(f"hetero and resname {code}")
                if het is not None:
                    seen = {(r.getChid(), int(r.getResnum())) for r in instances}
                    for r in het.getHierView().iterResidues():
                        if (r.getChid(), int(r.getResnum())) not in seen:
                            instances.append(r)
                if not instances:
                    stats["only_polymer_copies"] += 1
                    continue

                # One entry can hold several copies of the same ligand (e.g. a
                # homotetramer). Each copy sits in its own pocket, so each is a
                # separate training example rather than one averaged site.
                for res in instances:
                    inst = f"{res.getChid()}_{int(res.getResnum())}"
                    sel = st.select(
                        "chain {} and resname {} and resnum {}".format(
                            res.getChid(), code, int(res.getResnum())
                        )
                    )
                    pocket = pocket_residues(st, sel, cutoff=args.cutoff)
                    if pocket is None:
                        stats["pocket_empty"] += 1
                        continue

                    residues = [
                        (r.getChid(), int(r.getResnum()), r.getResname().strip().upper())
                        for r in pocket.getHierView().iterResidues()
                    ]
                    if len(residues) < MIN_POCKET_RESIDUES:
                        stats["pocket_too_small"] += 1
                        continue

                    chains = {}
                    for chid in sorted({c for c, _, _ in residues}):
                        if chid not in seq_cache:
                            seq_cache[chid] = chain_sequence(st, chid)
                        seq, index_of = seq_cache[chid]
                        idx = sorted(
                            index_of[num] for c, num, _ in residues
                            if c == chid and num in index_of
                        )
                        if not seq or not idx:
                            continue
                        chains[chid] = {"sequence": seq, "pocket_index": idx}
                    if not chains:
                        stats["no_mappable_chain"] += 1
                        continue

                    fh.write(json.dumps({
                        "pdb_id": pdb_id,
                        "ccd": code,
                        "instance": inst,
                        "families": e.get("families") or [],
                        "cats": e.get("cats") or [],
                        "uniprot": e.get("uniprot") or [],
                        "organism": e.get("sensor_organism") or "",
                        "method": e.get("method") or "",
                        "resolution": e.get("resolution"),
                        "cutoff": args.cutoff,
                        "n_pocket_residues": len(residues),
                        "residues": residues,
                        "chains": chains,
                    }) + "\n")
                    n_out += 1
                    stats["OK"] += 1

    print(f"T1 entries      {len(t1)}")
    print(f"pocket records  {n_out}")
    print("\nstats:")
    for k, v in stats.most_common():
        print(f"  {k:28s} {v:5d}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
