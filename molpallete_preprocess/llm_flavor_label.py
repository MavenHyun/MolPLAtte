#!/usr/bin/env python
"""Flavor-profile labelling with OpenRouter frontier LLMs, benchmarked first.

An LLM given a SMILES has no sensory measurement -- it can only infer flavor from
structural analogy. That makes its output a function of structure, which is the
same shape as every leak this project has already hit. So this script refuses to
be used blind: ``benchmark`` measures the model against compounds whose flavor is
KNOWN, and reports accuracy next to the baseline that matters.

The baseline is NOT random. NPClassifier class already predicts flavor to a
degree -- alkaloids are bitter 19% of the time against a 6% base rate. An LLM
that merely reproduces "alkaloid => bitter" has added nothing a deterministic
lookup could not. ``benchmark`` therefore scores the class-prior predictor on the
same compounds and prints both, so "did the LLM help?" is answerable.

Abstention is first-class. The vocabulary includes ``unknown``, and the prompt
says so explicitly: a model that cannot abstain will confabulate on the ~70% of
natural products nobody has ever tasted, and confident fabrication is worse than
a gap.

Usage
-----
    export OPENROUTER_API_KEY=...          # or ~/.config/openrouter.key
    python llm_flavor_label.py benchmark --n 300 --model anthropic/claude-opus-5
    python llm_flavor_label.py label --corpus <path> --model ... --out labels.jsonl
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

API = "https://openrouter.ai/api/v1/chat/completions"

#: Constrained output vocabulary. FlavorDB's raw set is 715 descriptors with 253
#: seen exactly once and 67% of the mass on sweet/sweet-like; asking a model to
#: hit that distribution is asking it to guess a long tail. These are the classes
#: with real support, plus the two that carry information by being absent.
VOCAB: Tuple[str, ...] = (
    # Taste primaries. sour/umami have almost no FlavorDB support (60 and 19
    # assignments) because FlavorDB is odor-centric, but a third of the
    # annotation targets are non-volatile compounds where taste is the ONLY
    # channel -- without these the model is forced to call them sweet/bitter.
    "sweet", "bitter", "sour", "salty", "umami",
    # Odor classes, ordered by FlavorDB support after synonym folding.
    "fruity", "green", "floral", "fatty", "woody", "spicy", "roasted",
    "sulfurous", "earthy", "nutty", "herbal", "medicinal", "citrus",
    "dairy", "alcoholic", "meaty", "minty",
    # Measured absence of odor -- a real class, distinct from not knowing.
    "odorless",
    # Explicit abstention. Load-bearing: without it the model confabulates on
    # the ~70% of natural products nobody has ever tasted.
    "unknown",
)

SYSTEM = (
    "You are a flavor chemist annotating compounds for a research dataset. "
    "You will be given molecules and must report their sensory profile.\n\n"
    "Rules:\n"
    f"1. Use ONLY these labels: {', '.join(VOCAB)}.\n"
    "2. Return at most 3 labels per compound, ordered by confidence.\n"
    "3. Use 'odorless' ONLY when you believe the compound is genuinely non-odorous "
    "(e.g. non-volatile, high molecular weight, highly polar).\n"
    "4. Use 'unknown' when you do not know. This is expected and correct for most "
    "natural products, which have never been sensorially characterised. Do NOT "
    "guess from structural analogy alone -- an 'unknown' is far more useful to us "
    "than a plausible invention.\n"
    "5. Give a confidence in [0,1] per compound reflecting how sure you are.\n\n"
    'Reply with JSON only: {"results":[{"id":"...","labels":["..."],'
    '"confidence":0.0}]}'
)


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
    if key:
        return key.strip()
    for p in (Path.home() / ".config" / "openrouter.key",
              Path.home() / ".openrouter.key"):
        if p.is_file():
            return p.read_text().strip()
    raise SystemExit(
        "No OpenRouter credential. Set OPENROUTER_API_KEY, or write the key to "
        "~/.config/openrouter.key (chmod 600)."
    )


def _post(payload: dict, key: str, retries: int = 4) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        API, data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "HTTP-Referer": "https://github.com/MolPallete",
                 "X-Title": "MolPallete flavor labelling"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code in (400, 401, 403):                 # not transient
                detail = ""
                try: detail = e.read().decode()[:300]
                except Exception: pass
                raise SystemExit(f"OpenRouter rejected the request ({code}): {detail}")
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 3)
    raise RuntimeError("unreachable")


def ask(model: str, items: Sequence[dict], key: str) -> Dict[str, dict]:
    """One call for a batch of compounds. Returns ``{id: {labels, confidence}}``."""
    lines = []
    for it in items:
        bits = [f'id={it["id"]}', f'SMILES={it["smiles"]}']
        if it.get("name"):
            bits.append(f'name={it["name"]}')
        if it.get("chem_class"):
            bits.append(f'class={it["chem_class"]}')
        if it.get("mw"):
            bits.append(f'MW={it["mw"]}')
        lines.append("  " + "; ".join(bits))
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Annotate these compounds:\n" + "\n".join(lines)},
        ],
    }
    resp = _post(payload, key)
    # A response can be malformed in three distinct ways, and all three must
    # degrade to "no annotation" rather than raise -- a 163K-call run cannot
    # abort on one bad reply. They are counted, not swallowed: an endpoint that
    # fails half its calls has to be VISIBLE. deepseek-v4-pro returned no
    # `choices` on 99 of 180 benchmark calls and the silent-{} version made that
    # look like poor abstention rather than a broken endpoint.
    if isinstance(resp, dict) and resp.get("error"):
        ask.n_error += 1
        return {}
    try:
        txt = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        ask.n_malformed += 1
        return {}
    if txt is None:
        # Reasoning models emit content=None when the token budget is consumed
        # by reasoning before any answer is produced.
        ask.n_empty += 1
        return {}
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1]
        txt = txt[4:] if txt.lower().startswith("json") else txt
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        s, e = txt.find("{"), txt.rfind("}")
        if s < 0 or e < 0:
            return {}
        try: data = json.loads(txt[s:e + 1])
        except json.JSONDecodeError: return {}
    out = {}
    for r in data.get("results", []):
        rid = str(r.get("id", "")).strip()
        if not rid:
            continue
        labs = [str(x).strip().lower() for x in (r.get("labels") or [])]
        out[rid] = {"labels": [x for x in labs if x in VOCAB],
                    "confidence": float(r.get("confidence") or 0.0)}
    return out


#: Failure counters. Read them after a run: a healthy endpoint leaves all three
#: at or near zero, and a large n_empty means max_tokens is being consumed by
#: reasoning tokens before any content is emitted.
ask.n_error = 0
ask.n_malformed = 0
ask.n_empty = 0


# --------------------------------------------------------------------------
# ground truth: FlavorDB labels joined onto COCONUT by InChIKey
# --------------------------------------------------------------------------

#: FlavorDB's raw descriptors mapped onto VOCAB. Anything unmapped is dropped
#: rather than guessed at, so the benchmark scores only what both sides can say.
_ALIAS = {
    "sweet-like": "sweet", "fruit": "fruity", "tropical": "fruity",
    "apple": "fruity", "pineapple": "fruity", "banana": "fruity",
    "berry": "fruity", "grape": "fruity", "peach": "fruity",
    "lemon": "citrus", "orange": "citrus", "rose": "floral",
    "balsam": "woody", "balsamic": "woody", "pine": "woody",
    "mint": "minty", "peppermint": "minty", "spearmint": "minty",
    "vegetable": "green", "grass": "green", "grassy": "green", "leafy": "green",
    "oily": "fatty", "creamy": "dairy", "buttery": "dairy", "cheesy": "dairy",
    "coffee": "roasted", "burnt": "roasted", "smoky": "roasted",
    "onion": "sulfurous", "garlic": "sulfurous", "sulfury": "sulfurous",
    "sweetbitter": "bitter", "fresh": "green", "mild": "unknown",
    "spice": "spicy", "pungent": "spicy", "nut": "nutty", "almond": "nutty",
}


def _norm(desc: str) -> Optional[str]:
    d = desc.strip().lower()
    if d in VOCAB:
        return d
    return _ALIAS.get(d)


def load_truth(flavordb: Path, coconut_csv: Path) -> List[dict]:
    """Compounds present in BOTH sources: structure + class + a measured label."""
    csv.field_size_limit(sys.maxsize)
    labs: Dict[str, str] = {}
    with (flavordb / "flavordb_molecules.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            cid = (r.get("cid") or "").strip()
            fp = (r.get("flavor_profile") or "").strip()
            if cid and fp:
                labs[cid] = fp
    by_ik: Dict[str, Set[str]] = {}
    with (flavordb / "properties.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            ik = (r.get("InChIKey") or "").strip()
            cid = (r.get("cid") or r.get("CID") or "").strip()
            if not ik or cid not in labs:
                continue
            mapped = {m for d in labs[cid].split(";") if (m := _norm(d))}
            if mapped:
                by_ik[ik] = mapped

    out: List[dict] = []
    with zipfile.ZipFile(coconut_csv) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as fh:
            for row in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8",
                                                       errors="replace")):
                ik = (row.get("standard_inchi_key") or "").strip()
                if ik not in by_ik:
                    continue
                out.append({
                    "id": row["identifier"].split(".")[0],
                    "smiles": (row.get("canonical_smiles") or "").strip(),
                    "name": (row.get("name") or "").strip(),
                    "chem_class": (row.get("np_classifier_pathway") or "").strip(),
                    "mw": (row.get("molecular_weight") or "").strip(),
                    "truth": by_ik[ik],
                })
    return [r for r in out if r["smiles"]]


def class_prior_predictor(train: Sequence[dict]) -> Dict[str, List[str]]:
    """The baseline the LLM has to beat: per-class most frequent descriptors."""
    by = collections.defaultdict(collections.Counter)
    glob = collections.Counter()
    for r in train:
        by[r["chem_class"] or "?"].update(r["truth"])
        glob.update(r["truth"])
    fallback = [d for d, _ in glob.most_common(3)]
    return {k: [d for d, _ in c.most_common(3)] or fallback for k, c in by.items()}, fallback


def score(pred: Dict[str, List[str]], items: Sequence[dict]) -> dict:
    tp = fp = fn = 0
    exact = abstain = 0
    for it in items:
        p = set(pred.get(it["id"], []))
        if p == {"unknown"} or not p:
            abstain += 1
            continue
        p.discard("unknown")
        t = it["truth"]
        tp += len(p & t); fp += len(p - t); fn += len(t - p)
        if p & t:
            exact += 1
    scored = len(items) - abstain
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(items), "answered": scored, "abstained": abstain,
        "abstain_rate": abstain / max(len(items), 1),
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "any_hit_rate": exact / max(scored, 1),
    }
