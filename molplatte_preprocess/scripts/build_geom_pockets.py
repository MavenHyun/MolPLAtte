"""Extract 3D pocket descriptors for the same 1,255 sites ESM-2 was scored on.

Uses the project's own ProDy parser so chain/resnum semantics match pockets.jsonl
exactly -- gemmi's auth/label naming does not, and silently finds nothing.

Four representations, chosen so a null is interpretable rather than ambiguous:

  comp    residue-type counts over the whole pocket. NO geometry at all. This is
          the control: whatever it scores is available from "which residues are
          nearby" without any 3D information.
  shell   the same counts split into distance shells from the ligand centroid
          (<5, 5-8, 8-12 A). comp plus radial geometry, so shell - comp isolates
          what the arrangement adds over the composition.
  shape   distances only, no residue identity: moments and a histogram of
          ligand-to-residue distances plus the pocket's inertia eigenvalues.
          Pure geometry.
  egnn    a randomly initialised E(n)-equivariant net over pocket Ca atoms.
          Literally the ablation asked for, but note it is UNTRAINED: a null
          here cannot separate "geometry is uninformative" from "these weights
          learned nothing", which is exactly why the three above are included.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/mogan/github/MolPLAtte/molplatte_preprocess/src")
from molplatte_prep.pocket_ligands import _parse_structure

CIF = Path("/home/mogan/datasets/tastepocket/structures/cif")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/geom_pockets.npz")
AA = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
      "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
AAI = {a: i for i, a in enumerate(AA)}
SHELLS = [(0,5),(5,8),(8,12)]

P=[json.loads(l) for l in open("/home/mogan/preprocessed/molplatte/tastepocket/pockets.jsonl")]
by_pdb={}
for p in P: by_pdb.setdefault(p["pdb_id"],[]).append(p)

keys=[]; comp=[]; shell=[]; shape=[]; coords=[]; restypes=[]
miss=0
for pdb, sites in sorted(by_pdb.items()):
    f=CIF/f"{pdb}.cif"
    if not f.exists(): miss+=len(sites); continue
    try: st=_parse_structure(f)
    except Exception: miss+=len(sites); continue
    for p in sites:
        ch,seq=p["instance"].split("_")
        lig=st.select(f"resname {p['ccd']} and chain {ch} and resnum {seq}")
        if lig is None: miss+=1; continue
        lc=lig.getCoords(); centre=lc.mean(0)
        want={(c,int(n)) for c,n,_ in p["residues"]}
        sel=st.select("protein and name CA")
        if sel is None: miss+=1; continue
        xyz=sel.getCoords(); chs=sel.getChids(); nums=sel.getResnums(); nms=sel.getResnames()
        keep=[i for i in range(len(xyz)) if (chs[i],int(nums[i])) in want]
        if len(keep)<8: miss+=1; continue
        X=xyz[keep]; names=[nms[i] for i in keep]
        # distance from each pocket CA to the NEAREST ligand atom
        d=np.linalg.norm(X[:,None,:]-lc[None,:,:],axis=2).min(1)
        c=np.zeros(20); sh=np.zeros(20*len(SHELLS))
        for nm,dd in zip(names,d):
            j=AAI.get(nm)
            if j is None: continue
            c[j]+=1
            for si,(lo,hi) in enumerate(SHELLS):
                if lo<=dd<hi: sh[si*20+j]+=1
        hist,_=np.histogram(d,bins=12,range=(0,15))
        Xc=X-X.mean(0)
        ev=np.linalg.eigvalsh(np.cov(Xc.T)) if len(Xc)>3 else np.zeros(3)
        sp=np.concatenate([hist,[d.mean(),d.std(),d.min(),d.max(),len(X)],np.sort(ev)[::-1]])
        keys.append(f"{p['pdb_id']}_{p['ccd']}_{p['instance']}")
        comp.append(c); shell.append(sh); shape.append(sp)
        coords.append((X-centre).astype(np.float32))
        restypes.append(np.array([AAI.get(n,20) for n in names],dtype=np.int64))
print(f"sites with 3D descriptors {len(keys):,}   missing {miss}")
np.savez_compressed(OUT, keys=np.array(keys), comp=np.array(comp),
                    shell=np.array(shell), shape=np.array(shape),
                    coords=np.array(coords,dtype=object),
                    restypes=np.array(restypes,dtype=object), allow_pickle=True)
print("wrote", OUT)
