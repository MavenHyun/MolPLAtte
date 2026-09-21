"""Merge zinc-10m + coconut-flavordb-full into one STEP 1 corpus, without duplicates.

Molecule ids cannot detect duplicates -- ZINC and COCONUT are different
namespaces -- so ZINC records whose WASHED CANONICAL SMILES already appears in
the flavour corpus are dropped, keeping the flavour copy (it carries the labels).
Tastepocket structures are dropped from BOTH sides: they are step 3's evaluation
set and must not be pretrained on.

Shards are hard-linked, not copied: both corpora use the same method, condvec
mode, graph-hash version and layout, so the payloads are byte-identical and a
link costs no disk.
"""
import glob, hashlib, json, os, shutil, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch

R = Path("/home/mogan/preprocessed/molplatte")
OUT = R / "zinc2020-coconut-flavordb-full" / "naveja_recap"
H = lambda s: hashlib.blake2b(s.encode(), digest_size=8).digest()


def probe(files):
    out = []
    for f in files:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
            if d.get("smiles"):
                out.append((f, d["mol_id"], H(d["smiles"])))
        except Exception:
            pass
    return out


def scan(corpus, workers=40, chunk=4000):
    fs = sorted(glob.glob(str(R / corpus / "naveja_recap" / "**" / "*.pt"), recursive=True))
    batches = [fs[i:i + chunk] for i in range(0, len(fs), chunk)]
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(probe, batches):
            res.extend(part)
    return res


if __name__ == "__main__":
    print("scanning tastepocket ...", flush=True)
    tp = {k for _, _, k in scan("tastepocket_corpus", workers=4, chunk=64)}
    print(f"  {len(tp)} structures reserved for step 3", flush=True)

    print("scanning flavour corpus ...", flush=True)
    fl = scan("coconut-flavordb-full")
    fl_keep = [(f, m, "coconut-flavordb-full") for f, m, k in fl if k not in tp]
    fl_keys = {k for _, _, k in fl}
    print(f"  flavour {len(fl):,} -> keep {len(fl_keep):,} "
          f"(dropped {len(fl)-len(fl_keep)} tastepocket structures)", flush=True)

    print("scanning zinc ...", flush=True)
    zn = scan("zinc-10m")
    zn_keep = [(f, m, "zinc-10m") for f, m, k in zn if k not in fl_keys and k not in tp]
    print(f"  zinc {len(zn):,} -> keep {len(zn_keep):,} "
          f"(dropped {len(zn)-len(zn_keep):,} duplicates)", flush=True)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    ids = []
    n = 0
    zinc_only_ids = []
    for src, mol_id, corpus in fl_keep + zn_keep:
        rel = Path(src).relative_to(R / corpus / "naveja_recap")
        if corpus == "zinc-10m":
            zinc_only_ids.append(mol_id)
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except FileExistsError:
            pass
        ids.append(mol_id)
        n += 1
        if n % 1_000_000 == 0:
            print(f"    linked {n:,}", flush=True)

    base = json.loads((R / "coconut-flavordb-full/naveja_recap/__meta__.json").read_text())
    zmeta = json.loads((R / "zinc-10m/naveja_recap/__meta__.json").read_text())
    base["ids"] = ids
    base["n_records"] = len(ids)
    base["sources"] = ["flavordb+coconut", "zinc"]
    base["source"] = "zinc2020+coconut-flavordb"
    base["merge"] = {
        "flavour_kept": len(fl_keep), "zinc_kept": len(zn_keep),
        "zinc_dropped_duplicate": len(zn) - len(zn_keep),
        "tastepocket_structures_excluded": len(tp),
        "note": "dedup on washed canonical SMILES; tastepocket reserved for step 3",
    }
    for k in ("n_decomps", "n_rgroups_per_decomp"):
        base.pop(k, None)
    base["zinc_only_ids"] = len(zinc_only_ids)
    (OUT / "__meta__.json").write_text(json.dumps(base))
    (OUT.parent / "zinc_only_molecule_ids.json").write_text(json.dumps(zinc_only_ids))
    shutil.copy(R / "coconut-flavordb-full/naveja_recap/__manifest__.json",
                OUT / "__manifest__.json")
    print(f"\n  merged corpus: {len(ids):,} molecules -> {OUT}")
