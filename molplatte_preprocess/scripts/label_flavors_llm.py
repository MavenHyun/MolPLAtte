#!/usr/bin/env python3
"""Resolve flavour labels for ligands the annotation tables do not cover.

Tier order is ``measured`` -> ``mined`` -> ``llm`` -> ``unknown``. The tier is
recorded per molecule so the LLM-labelled subset can be ablated out of any
result that depends on it.

THE PROMPT SEES THE MOLECULE ONLY -- name, formula, SMILES. Never the receptor
family, the PDB entry, or anything else about what the ligand was found bound
to. Telling the model "this binds TRPM8" would make it answer "cooling", and
the flavour half of the condvec would become a re-encoding of the pocket half:
conditioning would then show a large lift that means nothing, because both
halves would carry the same variable. Molecule identity is legitimate evidence;
binding partner is not.

``--control`` scores the model against ligands whose labels the measured table
already knows, blind. That number is what the ``llm`` tier is worth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from molplatte_prep.condvec import FLAVOR_LABELS  # noqa: E402

API = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-opus-5"
KEY_PATH = Path.home() / ".config" / "openrouter.key"

#: Below this the model's own labels are discarded in favour of ``unknown``.
#: A forced guess is worse than an abstention: ``unknown`` is a real bit in the
#: vector and the model is trained to read it.
MIN_CONFIDENCE = 0.45

#: `odorless` and `unknown` are bookkeeping, not sensory descriptors; the model
#: reaches them through `role` and `confidence` instead.
CHOOSABLE = [l for l in FLAVOR_LABELS if l not in ("odorless", "unknown")]

SYSTEM = f"""You are a flavour and fragrance chemist annotating molecules for a \
sensory model.

For the molecule given, report what it is and what a human perceives when it \
reaches a taste or olfactory receptor at a relevant concentration.

Reply with ONE JSON object, no prose, no code fence:
{{"identity": str, "role": str, "labels": [str], "confidence": float}}

- identity: what the molecule is, in a few words. "" if you do not recognise it.
- role: exactly one of
    "stimulus"  a tastant, odorant, sweetener or pheromone -- something \
perceived
    "modulator" a synthetic agonist/antagonist or drug-like compound with no \
sensory percept of its own
    "lipid"     a phospholipid, detergent, fatty-acyl or membrane structural \
component
    "other"     anything else, including cofactors, buffers and metabolites
- labels: zero or more of exactly these, lowercase, no others:
    {", ".join(CHOOSABLE)}
  Use [] when the molecule has no taste or odour, which is the correct answer \
for most "modulator" and "lipid" entries. Do not invent a percept from \
structure alone.
- confidence: 0.0-1.0, how sure you are of `labels`. Be honest; a low number is \
useful and a confident wrong label is not.

Judge the molecule on its own. You are not told what it binds to, and you \
should not speculate about that."""


def prompt_for(rec: dict) -> str:
    parts = [f"SMILES: {rec['smiles']}"]
    if rec.get("name"):
        parts.append(f"Name (PDB chemical component): {rec['name']}")
    if rec.get("formula"):
        parts.append(f"Formula: {rec['formula']}")
    if rec.get("mw"):
        parts.append(f"Molecular weight: {rec['mw']}")
    return "\n".join(parts)


def call(key: str, rec: dict, model: str, retries: int = 4) -> Optional[dict]:
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "max_tokens": 400,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt_for(rec)},
            ],
        }
    ).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                API,
                data=body,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.load(r)
            text = payload["choices"][0]["message"]["content"].strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(text)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                return None
        except (json.JSONDecodeError, KeyError, IndexError):
            return None
    return None


def normalise(raw: Optional[dict]) -> dict:
    """Clamp a model reply to the wire format. Anything odd degrades to unknown."""
    if not isinstance(raw, dict):
        return {"labels": [], "role": "", "identity": "", "confidence": 0.0,
                "note": "unparsable_reply"}
    labels = [l for l in (raw.get("labels") or []) if l in CHOOSABLE]
    try:
        conf = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    role = str(raw.get("role") or "")
    if role not in ("stimulus", "modulator", "lipid", "other"):
        role = "other"
    return {
        "labels": labels,
        "role": role,
        "identity": str(raw.get("identity") or "")[:200],
        "confidence": max(0.0, min(1.0, conf)),
        "note": "",
    }


def label_many(key: str, recs: List[dict], model: str, workers: int) -> Dict[str, dict]:
    def one(rec):
        return rec["ccd"], normalise(call(key, rec, model))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(one, recs))


def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ligands", type=Path, required=True)
    ap.add_argument("--annotations", type=Path,
                    default=Path.home() / "preprocessed/molplatte/annotations")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--control", action="store_true",
                    help="also score the model, blind, on table-known ligands")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    key = KEY_PATH.read_text().strip()

    def load_tbl(name):
        d = {}
        path = args.annotations / name
        for line in path.open():
            r = json.loads(line)
            ik = (r.get("inchikey") or "").strip()
            if ik:
                d.setdefault(ik, r)
        return d

    measured = load_tbl("flavor_measured.jsonl")
    mined = load_tbl("flavor_documented.jsonl")
    # Stereochemistry often differs between a CCD ideal SDF and a database
    # entry; the first InChIKey block is the constitution and matches anyway.
    meas_sk = {k.split("-")[0]: v for k, v in measured.items()}
    mine_sk = {k.split("-")[0]: v for k, v in mined.items()}

    ligands = [json.loads(l) for l in args.ligands.open()]
    kept = [r for r in ligands if r["ok"]]

    resolved: Dict[str, dict] = {}
    todo: List[dict] = []
    control: List[dict] = []
    for r in kept:
        ik = r.get("inchikey") or ""
        sk = ik.split("-")[0] if ik else ""
        hit = measured.get(ik) or (meas_sk.get(sk) if sk else None)
        if hit:
            resolved[r["ccd"]] = {"labels": hit["labels"], "source": "measured"}
            control.append(r)
            continue
        hit = mined.get(ik) or (mine_sk.get(sk) if sk else None)
        if hit:
            resolved[r["ccd"]] = {"labels": hit["labels"], "source": "mined"}
            continue
        todo.append(r)

    if args.limit:
        todo = todo[: args.limit]
        control = control[: args.limit]

    print(f"kept {len(kept)}   tables resolved {len(resolved)}   to label {len(todo)}")

    if args.control and control:
        print(f"\n== control: {len(control)} ligands the measured table knows, asked blind")
        got = label_many(key, control, args.model, args.workers)
        js, exact, any_overlap = [], 0, 0
        for r in control:
            truth = resolved[r["ccd"]]["labels"]
            pred = got[r["ccd"]]["labels"]
            j = jaccard(truth, pred)
            js.append(j)
            exact += set(truth) == set(pred)
            any_overlap += bool(set(truth) & set(pred))
        n = len(control)
        print(f"  mean Jaccard   {sum(js)/n:.3f}")
        print(f"  exact set      {exact}/{n}  ({100*exact/n:.0f}%)")
        print(f"  any overlap    {any_overlap}/{n}  ({100*any_overlap/n:.0f}%)")
        ctl_path = args.out.with_name(args.out.stem + "_control.jsonl")
        with ctl_path.open("w") as fh:
            for r in control:
                fh.write(json.dumps({"ccd": r["ccd"], "name": r["name"],
                                     "smiles": r["smiles"],
                                     "truth": resolved[r["ccd"]]["labels"],
                                     **got[r["ccd"]]}) + "\n")
        print(f"  wrote {ctl_path}")

    print(f"\n== labelling {len(todo)} unresolved with {args.model}")
    got = label_many(key, todo, args.model, args.workers)

    roles: Dict[str, int] = {}
    abstained = 0
    for r in todo:
        g = got[r["ccd"]]
        roles[g["role"]] = roles.get(g["role"], 0) + 1
        if g["labels"] and g["confidence"] >= MIN_CONFIDENCE:
            resolved[r["ccd"]] = {"labels": g["labels"], "source": "llm",
                                  "confidence": g["confidence"],
                                  "identity": g["identity"], "role": g["role"]}
        else:
            abstained += 1
            resolved[r["ccd"]] = {"labels": [], "source": "llm_abstain",
                                  "confidence": g["confidence"],
                                  "identity": g["identity"], "role": g["role"],
                                  "note": g["note"]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in kept:
            res = resolved.get(r["ccd"])
            if res is None:
                continue
            fh.write(json.dumps({"ccd": r["ccd"], "name": r["name"],
                                 "inchikey": r["inchikey"],
                                 "smiles": r["smiles"], **res}) + "\n")

    by_src: Dict[str, int] = {}
    for res in resolved.values():
        by_src[res["source"]] = by_src.get(res["source"], 0) + 1
    print("\nsources:")
    for k, v in sorted(by_src.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v:4d}")
    print("\nroles assigned to the unresolved set:")
    for k, v in sorted(roles.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v:4d}")
    print(f"\nabstained {abstained}/{len(todo)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
