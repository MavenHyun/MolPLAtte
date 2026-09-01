#!/usr/bin/env python
"""Recover compound names from PubChem for COCONUT entries that lack one.

COCONUT's CSV export names only 52.5% of compounds. The gap is a metadata
absence, not a statement about the molecule: CNP0217572 has no name in COCONUT
and is (Z,E)-trideca-4,7-dien-1-yl acetate, a documented lepidopteran pheromone.
Treating "unnamed" as "obscure" is therefore wrong, and it distorts anything
that reads the name field -- corpus inspection, EDA, annotation prompts.

Two lookups per compound, because they are COMPLEMENTARY rather than redundant.
Measured on 150 unnamed compounds:

    named by both            8.7%
    InChIKey only            1.3%
    SMILES only              1.3%
    in neither              88.7%

InChIKey is an exact hash, so any stereo or tautomer difference from PubChem's
normalisation returns 404; a SMILES structure lookup normalises differently and
recovers some of those (Apigenin-5-rhamnoside, Tambjamine H). It also misses
some the key finds, hence both.

Expected recovery is ~11% of 78,025 -- the ceiling is that the other 88.7% are
genuinely absent from PubChem, COCONUT having aggregated them from source
databases PubChem never ingested.

Rate limit: PubChem asks for <=5 requests/second. Default 4 workers respects
that. Resumable: re-running skips ids already in the output.
"""
from __future__ import annotations
import argparse, csv, json, re, sys, threading, time
import urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
#: Registry codes are not names. A compound whose only synonyms are these is
#: recorded as `code_only`, not as recovered -- otherwise the enrichment would
#: replace one meaningless identifier with another.
CODE = re.compile(
    r"^("
    r"SCHEMBL|ZINC|CID|AC1|CHEMBL|NSC|MFCD|DTXSID|UNII|EINECS|BDBM|SureCN|AKOS|"
    r"CS-|HY-|BP-|SY-|TC-|DB-|NCGC|SMR\d|SR-\d|Tox21|CCG-|EN\d{6}|Q\d{6,}|"
    r"\d{2,7}-\d{2}-\d|"          # CAS
    r"\d{3}-\d{3}-\d|"            # EINECS  (261-058-2 slipped past the CAS rule)
    r"[A-Z]{1,4}[- ]?\d{3,}$|"      # generic supplier/compound codes: BP-43400, SU-4942
    r"[A-Z0-9]{8,}$"
    r")", re.I)


#: Morphology of a chemical name: a functional-group suffix, or locants.
_CHEM_SUFFIX = re.compile(
    r"(yl|ate|ol|one|ene|ane|yne|ine|ide|oside|osid|oic|al|amide|amine|"
    r"anol|enol|ether|ester|acetate|benzo|phenyl|methyl|ethyl|glycer|chalcone)",
    re.I)
_LOCANT = re.compile(r"\d+[,'\-]")
_INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def _name_score(s: str):
    """Rank a synonym as a chemical NAME. Higher is better; None rejects.

    Selecting the SHORTEST synonym -- the obvious heuristic -- fails on real
    PubChem output, because registry codes are always shorter than names:

        RefChem:211291                       <- 13 chars, what shortest picks
        (Z,E)-Trideca-4,7-dien-1-yl acetate  <- 35 chars, the actual name

    Score by MORPHOLOGY instead: a chemical name carries a functional-group
    suffix or locants and is not digit-dense. Case is deliberately ignored --
    PubChem shouts some perfectly good names (4-CHLORO-2',6'-DIMETHOXYCHALCONE),
    so requiring lowercase would discard them.
    """
    if not (3 < len(s) < 160):
        return None
    if ":" in s or _INCHIKEY.match(s) or CODE.match(s):
        return None
    letters = sum(c.isalpha() for c in s)
    if letters < 4 or sum(c.isdigit() for c in s) > letters:
        return None
    score = 0
    if _CHEM_SUFFIX.search(s):
        score += 3
    if _LOCANT.search(s):
        score += 2
    if any(c.islower() for c in s[1:]):
        score += 1
    return score if score >= 3 else None


def _looks_chemical(s: str) -> bool:
    return _name_score(s) is not None


def _synonyms(url: str, data: bytes | None = None):
    try:
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"} if data else {})
        with urllib.request.urlopen(req, timeout=25) as fh:
            d = json.loads(fh.read())
        return d["InformationList"]["Information"][0].get("Synonym", []) or []
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
            OSError, ValueError, KeyError, IndexError):
        return []


def best_name(syns):
    """Highest-scoring synonym; among ties the shortest, i.e. the common name."""
    scored = [(sc, x) for x in syns if (sc := _name_score(x)) is not None]
    if not scored:
        return None
    top = max(sc for sc, _ in scored)
    return min((x for sc, x in scored if sc == top), key=len)


def resolve(rec):
    syn = _synonyms(f"{BASE}/inchikey/{rec['inchikey']}/synonyms/JSON") if rec.get("inchikey") else []
    via = "inchikey"
    if not best_name(syn) and rec.get("smiles"):
        s2 = _synonyms(f"{BASE}/smiles/synonyms/JSON",
                       data=urllib.parse.urlencode({"smiles": rec["smiles"]}).encode())
        if best_name(s2):
            syn, via = s2, "smiles"
    nm = best_name(syn)
    return {"id": rec["id"], "name": nm,
            "via": via if nm else None,
            "status": "named" if nm else ("code_only" if syn else "missing"),
            "n_synonyms": len(syn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="docs/flavor_annotation_targets.csv")
    ap.add_argument("--out", default="pubchem_names.jsonl")
    ap.add_argument("--workers", type=int, default=4, help="PubChem asks for <=5 req/s")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    csv.field_size_limit(sys.maxsize)

    rows = [r for r in csv.DictReader(open(a.targets))
            if not (r.get("name") or "").strip() and (r.get("inchikey") or "").strip()]
    out = Path(a.out)
    done = set()
    if out.exists():
        for line in out.open():
            try: done.add(json.loads(line)["id"])
            except Exception: pass
    todo = [r for r in rows if r["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[pubchem] {len(rows):,} unnamed, {len(done):,} resolved already, "
          f"{len(todo):,} to do ({a.workers} workers)", flush=True)

    lock = threading.Lock(); n = named = 0; t0 = time.time()
    fh = out.open("a")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(resolve, r) for r in todo]):
            r = f.result()
            with lock:
                fh.write(json.dumps(r) + "\n"); n += 1
                named += r["status"] == "named"
                if n % 1000 == 0:
                    fh.flush()
                    rate = n / max(time.time() - t0, 1)
                    print(f"  {n:,}/{len(todo):,}  named {named:,} ({named/n:.1%})  "
                          f"{rate:.1f}/s  eta {(len(todo)-n)/max(rate,.01)/3600:.1f}h",
                          flush=True)
    fh.close()
    print(f"[pubchem] done: {n:,} looked up, {named:,} named ({named/max(n,1):.1%})", flush=True)


if __name__ == "__main__":
    main()
