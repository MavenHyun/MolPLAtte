#!/usr/bin/env python3
"""Does nearest-pocket R-group retrieval beat just predicting common fragments?

The granularity probe showed pocket similarity predicts R-groups at 10-20 sigma
over a RANDOM-POCKET baseline. But the tastepocket targets turned out to sit in
the HEAD of the frequency distribution -- 0.286% of library entries carrying
48.6% of the prior mass, typical target seen ~1,200 times in 393k records. So a
constant "guess the commonest fragments" predictor is already strong, and
enrichment over random does not establish that the pocket adds anything usable.

Three predictors, identical budget k, identical candidate space:

  FREQ     the k globally most frequent R-groups. A CONSTANT prediction --
           identical for every query, uses no pocket at all.
  RANDPKT  union of m RANDOM pockets' R-groups, grown until k hashes. Matched
           to NNPKT in generative process and therefore in frequency profile;
           differs only in that the pockets are not chosen by similarity.
  NNPKT    union of the m NEAREST pockets by ESM-2 cosine, same growth rule.

  NNPKT vs RANDPKT isolates pocket SIMILARITY at matched frequency.
  NNPKT vs FREQ     asks whether the pocket is worth consulting at all.

Neighbours are always restricted to a different receptor AND a different
ligand, as in every other pocket probe.
"""
from __future__ import annotations
import glob, gzip, json, pickle, random, sys
from collections import defaultdict

import numpy as np
import torch

D = "/home/mogan/preprocessed/molplatte/tastepocket"
CORPUS = "/home/mogan/preprocessed/molplatte/tastepocket_corpus/naveja_recap"
VOCAB = ("/home/mogan/preprocessed/molplatte/union_vocab/"
         "base-full__crossdocked__tastepocket/rgroup_vocab.pkl.gz")
KS = [1, 5, 10, 25, 50, 100]

with gzip.open(VOCAB, "rb") as fh:
    ent = pickle.load(fh)["entries"]
count = {h: e["count"] for h, e in ent.items()}
freq_rank = [h for h, _ in sorted(count.items(), key=lambda kv: -kv[1])]

mol_hashes = {}
for f in sorted(glob.glob(f"{CORPUS}/**/*.pt", recursive=True)):
    d = torch.load(f, map_location="cpu", weights_only=False)
    hs = set()
    for dec in d["decompositions"]:
        hs.update(dec["rgroup_hashes"])
    mol_hashes[d["mol_id"]] = hs

esm = np.load(f"{D}/pocket_esm2_650M.npz", allow_pickle=True)
E = {str(k): v for k, v in zip(esm["keys"], esm["embeddings"])}
pk = {f"{p['pdb_id']}_{p['ccd']}_{p['instance']}": p
      for p in map(json.loads, open(f"{D}/pockets.jsonl"))}


def mol_id_of(p):
    up = p["uniprot"]
    return f"{p['ccd']}__{';'.join(up) if isinstance(up, list) else up}"


keys = [k for k in pk if k in E and mol_id_of(pk[k]) in mol_hashes
        and mol_hashes[mol_id_of(pk[k])]]
mids = [mol_id_of(pk[k]) for k in keys]
print(f"queries: {len(keys)}")

M = np.stack([E[k] for k in keys]).astype(np.float64)
M = M - M.mean(0); sd = M.std(0); sd[sd == 0] = 1; M = M / sd
M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
S = M @ M.T
unis = np.array([str(pk[k]["uniprot"]) for k in keys])
ccds = np.array([pk[k]["ccd"] for k in keys])


def grow(order, k):
    """Union R-groups of pockets in `order` until k distinct hashes."""
    out = []
    seen = set()
    for j in order:
        for h in sorted(mol_hashes[mids[j]], key=lambda h: -count.get(h, 0)):
            if h not in seen:
                seen.add(h); out.append(h)
                if len(out) >= k:
                    return out
    return out


rng = random.Random(0)
hit = {p: {k: [] for k in KS} for p in ("FREQ", "RANDPKT", "NNPKT")}
rec = {p: {k: [] for k in KS} for p in ("FREQ", "RANDPKT", "NNPKT")}

for i in range(len(keys)):
    true = mol_hashes[mids[i]]
    m = (unis != unis[i]) & (ccds != ccds[i])
    if not m.any():
        continue
    cand = np.where(m)[0]
    nn_order = cand[np.argsort(-S[i, cand])]
    rnd_order = np.array(rng.sample(list(cand), len(cand)))
    for k in KS:
        preds = {"FREQ": freq_rank[:k],
                 "RANDPKT": grow(rnd_order, k),
                 "NNPKT": grow(nn_order, k)}
        for p, pred in preds.items():
            s = set(pred)
            hit[p][k].append(1.0 if s & true else 0.0)
            rec[p][k].append(len(s & true) / len(true))

print(f"\n{'k':>5}" + "".join(f"{p:>12}" for p in ("FREQ", "RANDPKT", "NNPKT"))
      + f"{'NN-RAND':>11}{'σ':>7}{'NN-FREQ':>11}{'σ':>7}")
print("hit@k  (query has >=1 true R-group in the prediction)")
for k in KS:
    f_, r_, n_ = (np.array(hit[p][k]) for p in ("FREQ", "RANDPKT", "NNPKT"))
    d1, d2 = n_ - r_, n_ - f_
    se = lambda v: v.std() / np.sqrt(len(v)) if v.std() > 0 else float("nan")
    print(f"{k:>5}{f_.mean():>12.4f}{r_.mean():>12.4f}{n_.mean():>12.4f}"
          f"{d1.mean():>+11.4f}{d1.mean()/se(d1):>7.1f}{d2.mean():>+11.4f}{d2.mean()/se(d2):>7.1f}")
print("recall@k  (fraction of the query's true R-groups recovered)")
for k in KS:
    f_, r_, n_ = (np.array(rec[p][k]) for p in ("FREQ", "RANDPKT", "NNPKT"))
    d1, d2 = n_ - r_, n_ - f_
    se = lambda v: v.std() / np.sqrt(len(v)) if v.std() > 0 else float("nan")
    print(f"{k:>5}{f_.mean():>12.4f}{r_.mean():>12.4f}{n_.mean():>12.4f}"
          f"{d1.mean():>+11.4f}{d1.mean()/se(d1):>7.1f}{d2.mean():>+11.4f}{d2.mean()/se(d2):>7.1f}")
