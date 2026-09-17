"""Score typed non-covalent interaction descriptors on the ESM-2 probe.

Protocol is IDENTICAL to egnn_pocket_probe.py -- for each held-out pocket take
the nearest OTHER pocket by cosine, restricted to a different receptor AND a
different ligand, and compare its ligand to the true one by Morgan/Tanimoto.
The delta against a randomly drawn pocket from the same candidate set is the
signal. Cross-family repeats it with the neighbour also forced into a different
receptor family.

Read the apo/holo distinction in build_ifp_pockets.py before reading the table:
the holo rows are ORACLES that saw the answer, and are reported only as a
ceiling.
"""
import json, random, sys
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, DataStructs
RDLogger.DisableLog("rdApp.*")

D = "/home/mogan/preprocessed/molplatte/tastepocket"
IFP = np.load("/tmp/ifp_pockets.npz", allow_pickle=True)
esm = np.load(f"{D}/pocket_esm2_650M.npz", allow_pickle=True)
E = {str(k): v for k, v in zip(esm["keys"], esm["embeddings"])}
lig = {j["ccd"]: j for j in map(json.loads, open(f"{D}/ligands.jsonl"))}
pk = {f"{p['pdb_id']}_{p['ccd']}_{p['instance']}": p
      for p in map(json.loads, open(f"{D}/pockets.jsonl"))}
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp = {c: gen.GetFingerprint(Chem.MolFromSmiles(j["smiles"]))
      for c, j in lig.items() if j.get("smiles") and Chem.MolFromSmiles(j["smiles"])}

ifp_keys = [str(k) for k in IFP["keys"]]
keys = [k for k in ifp_keys if k in E and k in pk]
sel = [ifp_keys.index(k) for k in keys]
print(f"pockets: {len(ifp_keys)} with IFP, {len(keys)} also having ESM-2")

# The September geometry descriptors, fed through THIS harness. They are the
# control: if they do not reproduce the published +0.2177 / +0.0325, the probe
# code is wrong and no other row in the table means anything.
GEO = np.load("/tmp/geom_pockets.npz", allow_pickle=True)
gkeys = [str(k) for k in GEO["keys"]]
gsel = {k: i for i, k in enumerate(gkeys)}
gidx = [gsel[k] for k in keys if k in gsel]
has_geo = [k in gsel for k in keys]
assert all(has_geo), f"{has_geo.count(False)} probe keys absent from geom_pockets.npz"

reps = {
    "ESM-2 1280 (sequence)":          np.stack([E[k] for k in keys]),
    "shell  PUBLISHED (control)":     GEO["shell"][gidx],
    "comp   PUBLISHED (control)":     GEO["comp"][gidx],
    "shape  PUBLISHED (control)":     GEO["shape"][gidx],
    "apo-shell  (restype x dist)":    IFP["apo_shell"][sel],
    "apo-pharm  (pharmacophore)":     IFP["apo_pharm"][sel],
    "apo  shell+pharm":               np.hstack([IFP["apo_shell"][sel], IFP["apo_pharm"][sel]]),
    "holo-itype (ORACLE)":            IFP["holo_itype"][sel],
    "holo-ifp   (ORACLE)":            IFP["holo_ifp"][sel],
    "holo-ifp + ESM-2 (ORACLE)":      np.hstack([np.stack([E[k] for k in keys]), IFP["holo_ifp"][sel]]),
}
# the question that matters: does typed NCI information ADD anything on top of
# the representations that already work?
_esm = np.stack([E[k] for k in keys])
reps["shell + apo-pharm"]            = np.hstack([GEO["shell"][gidx], IFP["apo_pharm"][sel]])
reps["ESM-2 + apo-pharm"]            = np.hstack([_esm, IFP["apo_pharm"][sel]])
reps["ESM-2 + holo-itype (ORACLE)"]  = np.hstack([_esm, IFP["holo_itype"][sel]])

unis = np.array([str(pk[k]["uniprot"]) for k in keys])
fams = np.array([str(pk[k]["families"]) for k in keys])
ccds = np.array([pk[k]["ccd"] for k in keys])
ok_row = np.array([c in fp for c in ccds])


def probe(M, label):
    M = np.asarray(M, dtype=np.float64)
    M = M - M.mean(0); sd = M.std(0); sd[sd == 0] = 1; M = M / sd
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = M @ M.T; rng = random.Random(0)
    nn_, rd_, nnx, rdx = [], [], [], []
    for i in range(len(keys)):
        if not ok_row[i]: continue
        m = (unis != unis[i]) & (ccds != ccds[i]) & ok_row
        if not m.any(): continue
        c = np.where(m)[0]; j = c[np.argmax(S[i, c])]
        nn_.append(DataStructs.TanimotoSimilarity(fp[ccds[i]], fp[ccds[j]]))
        rd_.append(DataStructs.TanimotoSimilarity(fp[ccds[i]], fp[ccds[rng.choice(list(c))]]))
        xf = c[fams[c] != fams[i]]
        if len(xf):
            k = xf[np.argmax(S[i, xf])]
            nnx.append(DataStructs.TanimotoSimilarity(fp[ccds[i]], fp[ccds[k]]))
            rdx.append(DataStructs.TanimotoSimilarity(fp[ccds[i]], fp[ccds[rng.choice(list(xf))]]))
    d = np.array(nn_) - np.array(rd_); dx = np.array(nnx) - np.array(rdx)
    se = lambda v: v.std() / np.sqrt(len(v))
    print(f"  {label:<30}{np.mean(nn_):>8.4f}{np.mean(rd_):>8.4f}{d.mean():>+9.4f}"
          f"{d.mean()/se(d):>7.1f}{dx.mean():>+11.4f}{dx.mean()/se(dx):>7.1f}")
    return d.mean(), dx.mean()


print(f"\n{'representation':<32}{'nearest':>8}{'random':>8}{'delta':>9}{'SE':>7}"
      f"{'xfam delta':>11}{'SE':>7}")
res = {k: probe(v, k) for k, v in reps.items()}
b = res["ESM-2 1280 (sequence)"]
print(f"\nbar to clear: unseen-ligand {b[0]:+.4f}, cross-family {b[1]:+.4f}")
print("published `shell` control: unseen-ligand +0.2177, cross-family +0.0325")
for k, (a, x) in res.items():
    if k.startswith("ESM"): continue
    v = "CLEARS" if (a > b[0] and x > b[1]) else ("partial" if (a > b[0] or x > b[1]) else "below")
    tag = "  [oracle]" if "ORACLE" in k else ""
    print(f"  {k:<30} {v}{tag}")

# --- robustness: 283 pockets used PDB-perceived bonds because the deposited
# ligand is incomplete. If the conclusions move when they are dropped, the
# degraded rows were driving them.
deg = set(str(k) for k in IFP["degraded"])
clean = [i for i, k in enumerate(keys) if k not in deg]
print(f"\n--- clean-only robustness check ({len(clean)}/{len(keys)} pockets, "
      f"{len(keys)-len(clean)} degraded dropped)")
_k_all = keys
for name in ("ESM-2 1280 (sequence)", "shell  PUBLISHED (control)",
             "apo  shell+pharm", "holo-ifp   (ORACLE)", "holo-ifp + ESM-2 (ORACLE)"):
    M = reps[name][clean]
    keys = [_k_all[i] for i in clean]
    unis_c, fams_c, ccds_c = unis[clean], fams[clean], ccds[clean]
    g_unis, g_fams, g_ccds, g_ok = unis, fams, ccds, ok_row
    unis, fams, ccds, ok_row = unis_c, fams_c, ccds_c, g_ok[clean]
    probe(M, name)
    keys = _k_all
    unis, fams, ccds, ok_row = g_unis, g_fams, g_ccds, g_ok
