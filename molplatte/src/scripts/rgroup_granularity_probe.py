#!/usr/bin/env python3
"""Do pockets predict the R-GROUP, or only the whole ligand?

Every pocket probe so far asked: does pocket similarity predict LIGAND
similarity? ESM-2 answers +0.2365 -- pockets clearly carry ligand-level signal.
But the model's retrieval target is not a ligand, it is an R-GROUP keyed by a WL
subgraph hash. If pocket similarity predicts the whole ligand but NOT its
R-groups, then the finetuning null is a property of the TARGET, not of the data
volume -- and no amount of extra tastepocket records would fix it.

Protocol is identical to egnn_pocket_probe.py / ifp_pocket_probe.py: for each
held-out pocket take the nearest OTHER pocket by cosine, restricted to a
different receptor AND a different ligand; score the neighbour's molecule
against the true one; compare against a randomly drawn pocket from the same
candidate set. Only the SCORING TARGET changes:

  ligand     Morgan/Tanimoto on the whole ligand   (the published control)
  core       Morgan/Tanimoto on the decomposition core (scaffold)
  rgroup-tan best-match mean Tanimoto over R-group fragments (soft)
  rgroup-jac Jaccard over WL hash SETS -- the model's EXACT target granularity

Reading it:
  ligand high, core high, rgroup ~0  -> the pocket fixes the scaffold and says
                                        nothing about the decoration. The
                                        retrieval target is the problem.
  rgroup tracks ligand               -> the target is fine and the null is data.
"""
from __future__ import annotations
import glob, json, random, statistics as st, sys
from collections import defaultdict

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, DataStructs
RDLogger.DisableLog("rdApp.*")

D = "/home/mogan/preprocessed/molplatte/tastepocket"
CORPUS = "/home/mogan/preprocessed/molplatte/tastepocket_corpus/naveja_recap"

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp_of(smiles):
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    return gen.GetFingerprint(m) if m is not None else None


def rgroup_smiles(mol, atoms):
    """SMILES of the R-group fragment defined by its atom indices."""
    try:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), canonical=True)
    except Exception:
        return None


print("loading corpus decompositions ...")
mol = {}                      # mol_id -> dict(ligand_fp, core_fps, rg_fps, hashes)
for f in sorted(glob.glob(f"{CORPUS}/**/*.pt", recursive=True)):
    d = torch.load(f, map_location="cpu", weights_only=False)
    mid = d["mol_id"]
    parent = Chem.MolFromSmiles(d["smiles"]) if d.get("smiles") else None
    if parent is None:
        continue
    hashes, rg_fps, core_fps = set(), [], []
    for dec in d["decompositions"]:
        hashes.update(dec["rgroup_hashes"])
        # core_smiles is unpopulated in this corpus (empty for all 734
        # decompositions) while core_atoms is present -- rebuild from indices,
        # otherwise the core row silently scores n=0.
        cs = (dec.get("core_smiles") or "").strip()
        if not cs and dec.get("core_atoms"):
            cs = rgroup_smiles(parent, dec["core_atoms"])
        cf = fp_of(cs)
        if cf is not None:
            core_fps.append(cf)
        for rg in dec["rgroups"]:
            s = rgroup_smiles(parent, rg["rgroup_atoms"])
            f_ = fp_of(s)
            if f_ is not None:
                rg_fps.append(f_)
    mol[mid] = {"lig": gen.GetFingerprint(parent), "core": core_fps,
                "rg": rg_fps, "hash": hashes}
print(f"  molecules with decompositions: {len(mol)}")

esm = np.load(f"{D}/pocket_esm2_650M.npz", allow_pickle=True)
E = {str(k): v for k, v in zip(esm["keys"], esm["embeddings"])}
pk = {f"{p['pdb_id']}_{p['ccd']}_{p['instance']}": p
      for p in map(json.loads, open(f"{D}/pockets.jsonl"))}

# pocket key -> corpus mol_id  ("CCD__UNIPROT;UNIPROT")
def mol_id_of(p):
    up = p["uniprot"]
    up = ";".join(up) if isinstance(up, list) else str(up)
    return f"{p['ccd']}__{up}"

keys = [k for k in pk if k in E and mol_id_of(pk[k]) in mol]
print(f"  pockets joined to a decomposition: {len(keys)} / {len(pk)}")
mids = [mol_id_of(pk[k]) for k in keys]

M = np.stack([E[k] for k in keys]).astype(np.float64)
M = M - M.mean(0); sd = M.std(0); sd[sd == 0] = 1; M = M / sd
M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
S = M @ M.T

unis = np.array([str(pk[k]["uniprot"]) for k in keys])
fams = np.array([str(pk[k]["families"]) for k in keys])
ccds = np.array([pk[k]["ccd"] for k in keys])

T = DataStructs.TanimotoSimilarity


def best_match(a, b):
    """Symmetric mean of best-partner Tanimoto between two fragment sets."""
    if not a or not b:
        return None
    f = lambda x, y: st.fmean(max(T(p, q) for q in y) for p in x)
    return 0.5 * (f(a, b) + f(b, a))


def jaccard(a, b):
    if not a and not b:
        return None
    u = len(a | b)
    return len(a & b) / u if u else None


def target(i, j, kind):
    A, B = mol[mids[i]], mol[mids[j]]
    if kind == "ligand":
        return T(A["lig"], B["lig"])
    if kind == "core":
        return best_match(A["core"], B["core"])
    if kind == "rgroup-tan":
        return best_match(A["rg"], B["rg"])
    if kind == "rgroup-jac":
        return jaccard(A["hash"], B["hash"])
    raise ValueError(kind)


KINDS = ["ligand", "core", "rgroup-tan", "rgroup-jac"]
print(f"\n{'target':<14}{'nearest':>9}{'random':>9}{'delta':>10}{'SE':>7}"
      f"{'xfam delta':>12}{'SE':>7}{'n':>6}")
summary = {}
for kind in KINDS:
    rng = random.Random(0)
    nn_, rd_, nnx, rdx = [], [], [], []
    for i in range(len(keys)):
        m = (unis != unis[i]) & (ccds != ccds[i])
        if not m.any():
            continue
        c = np.where(m)[0]
        j = c[int(np.argmax(S[i, c]))]
        a = target(i, j, kind)
        b = target(i, int(rng.choice(list(c))), kind)
        if a is not None and b is not None:
            nn_.append(a); rd_.append(b)
        xf = c[fams[c] != fams[i]]
        if len(xf):
            k2 = xf[int(np.argmax(S[i, xf]))]
            a2 = target(i, k2, kind)
            b2 = target(i, int(rng.choice(list(xf))), kind)
            if a2 is not None and b2 is not None:
                nnx.append(a2); rdx.append(b2)
    d = np.array(nn_) - np.array(rd_)
    dx = np.array(nnx) - np.array(rdx)
    se = lambda v: v.std() / np.sqrt(max(len(v), 1))
    print(f"{kind:<14}{np.mean(nn_):>9.4f}{np.mean(rd_):>9.4f}{d.mean():>+10.4f}"
          f"{d.mean()/se(d):>7.1f}{dx.mean():>+12.4f}{dx.mean()/se(dx):>7.1f}{len(d):>6}")
    summary[kind] = (d.mean(), d.mean() / se(d), dx.mean(), dx.mean() / se(dx))

lg = summary["ligand"]
print(f"\nligand-level signal: {lg[0]:+.4f} ({lg[1]:+.1f}σ)")
for kind in KINDS[1:]:
    v = summary[kind]
    frac = v[0] / lg[0] if lg[0] else float("nan")
    print(f"  {kind:<12} {v[0]:+.4f} ({v[1]:+.1f}σ)   = {100*frac:5.1f}% of the ligand-level delta")
