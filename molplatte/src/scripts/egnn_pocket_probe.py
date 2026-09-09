"""Score 3D pocket representations on the same probe ESM-2 was scored on.

Protocol is identical to pocket_probe2.py: for each held-out pocket take the
nearest OTHER pocket by cosine, restricted to a different receptor AND a
different ligand, and compare its ligand to the true one by Morgan/Tanimoto.
The delta against a randomly drawn pocket from the same candidate set is the
signal. Cross-family repeats it with the neighbour also forced into a different
receptor family.
"""
import json, random, sys
import numpy as np, torch, torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, DataStructs
RDLogger.DisableLog("rdApp.*")

D="/home/mogan/preprocessed/molplatte/tastepocket"
G=np.load("/tmp/geom_pockets.npz",allow_pickle=True)
esm=np.load(f"{D}/pocket_esm2_650M.npz",allow_pickle=True)
E={str(k):v for k,v in zip(esm["keys"],esm["embeddings"])}
lig={j["ccd"]:j for j in map(json.loads,open(f"{D}/ligands.jsonl"))}
pk={f"{p['pdb_id']}_{p['ccd']}_{p['instance']}":p
    for p in map(json.loads,open(f"{D}/pockets.jsonl"))}
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
fp={c:gen.GetFingerprint(Chem.MolFromSmiles(j["smiles"]))
    for c,j in lig.items() if j.get("smiles") and Chem.MolFromSmiles(j["smiles"])}

# ---- random-weight EGNN over pocket CA atoms -----------------------------
class EGNNLayer(nn.Module):
    """One E(n)-equivariant message-passing layer (Satorras et al. 2021).

    Messages depend on coordinates only through squared distance, so features
    are invariant and positions update along pairwise differences -- the layer
    is equivariant to rotation and translation by construction.
    """
    def __init__(s,d):
        super().__init__()
        s.edge=nn.Sequential(nn.Linear(2*d+1,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU())
        s.node=nn.Sequential(nn.Linear(2*d,d),nn.SiLU(),nn.Linear(d,d))
    def forward(s,h,x):
        n=h.shape[0]
        diff=x[:,None,:]-x[None,:,:]
        d2=(diff**2).sum(-1,keepdim=True)
        hi=h[:,None,:].expand(n,n,-1); hj=h[None,:,:].expand(n,n,-1)
        m=s.edge(torch.cat([hi,hj,d2],-1))
        mask=(1-torch.eye(n,device=h.device)).unsqueeze(-1)
        agg=(m*mask).sum(1)
        return h+s.node(torch.cat([h,agg],-1))

def egnn_embed(coords,restypes,dim=64,layers=3,seed=0,cap=64):
    torch.manual_seed(seed)
    emb=nn.Embedding(21,dim); net=nn.ModuleList([EGNNLayer(dim) for _ in range(layers)])
    for p in list(emb.parameters())+list(net.parameters()): p.requires_grad_(False)
    out=[]
    with torch.no_grad():
        for X,R in zip(coords,restypes):
            if len(X)>cap:
                idx=np.argsort(np.linalg.norm(X,axis=1))[:cap]; X,R=X[idx],R[idx]
            x=torch.tensor(np.asarray(X),dtype=torch.float32)
            h=emb(torch.tensor(np.asarray(R)))
            for l in net: h=l(h,x)
            out.append(h.mean(0).numpy())
    return np.stack(out)

keys=[str(k) for k in G["keys"]]
print("building random-weight EGNN embeddings ...")
egnn=egnn_embed(G["coords"],G["restypes"])

reps={"ESM-2 1280 (sequence)":np.stack([E[k] for k in keys]),
      "comp   (residues, NO geometry)":G["comp"],
      "shell  (residues x distance)":G["shell"],
      "shape  (geometry, no identity)":G["shape"],
      "EGNN   (random weights)":egnn}

unis=np.array([str(pk[k]["uniprot"]) for k in keys])
fams=np.array([str(pk[k]["families"]) for k in keys])
ccds=np.array([pk[k]["ccd"] for k in keys])
ok_row=np.array([c in fp for c in ccds])

def probe(M,label):
    M=np.asarray(M,dtype=np.float64)
    M=M-M.mean(0); sd=M.std(0); sd[sd==0]=1; M=M/sd          # scale-free comparison
    M=M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)
    S=M@M.T; rng=random.Random(0)
    nn_,rd_,nnx,rdx=[],[],[],[]
    for i in range(len(keys)):
        if not ok_row[i]: continue
        m=(unis!=unis[i])&(ccds!=ccds[i])&ok_row
        if not m.any(): continue
        c=np.where(m)[0]; j=c[np.argmax(S[i,c])]
        nn_.append(DataStructs.TanimotoSimilarity(fp[ccds[i]],fp[ccds[j]]))
        rd_.append(DataStructs.TanimotoSimilarity(fp[ccds[i]],fp[ccds[rng.choice(list(c))]]))
        xf=c[fams[c]!=fams[i]]
        if len(xf):
            k=xf[np.argmax(S[i,xf])]
            nnx.append(DataStructs.TanimotoSimilarity(fp[ccds[i]],fp[ccds[k]]))
            rdx.append(DataStructs.TanimotoSimilarity(fp[ccds[i]],fp[ccds[rng.choice(list(xf))]]))
    d=np.array(nn_)-np.array(rd_); dx=np.array(nnx)-np.array(rdx)
    se=lambda v: v.std()/np.sqrt(len(v))
    print(f"  {label:<32}{np.mean(nn_):>8.4f}{np.mean(rd_):>8.4f}{d.mean():>+9.4f}"
          f"{d.mean()/se(d):>7.1f}{dx.mean():>+11.4f}{dx.mean()/se(dx):>7.1f}")
    return d.mean(), dx.mean()

print(f"\n{'representation':<34}{'nearest':>8}{'random':>8}{'delta':>9}{'SE':>7}{'xfam delta':>11}{'SE':>7}")
res={k:probe(v,k) for k,v in reps.items()}
b=res["ESM-2 1280 (sequence)"]
print(f"\nbar to clear: unseen-ligand {b[0]:+.4f}, cross-family {b[1]:+.4f}")
for k,(a,x) in res.items():
    if k.startswith("ESM"): continue
    v="CLEARS" if (a>b[0] and x>b[1]) else ("partial" if (a>b[0] or x>b[1]) else "below")
    print(f"  {k:<34} {v}")
