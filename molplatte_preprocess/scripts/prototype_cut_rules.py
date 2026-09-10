"""How large can R-groups get under different cut rules?

The current pipeline yields R-groups at 6-18% of the parent regardless of parent
size, because only NON-RING single bonds are cut and the ring system always
lands in the core -- so what detaches is whatever peripheral substituent hangs
off it. This measures what other rules would give, on the same molecules.

  A current      naveja_recap as configured (ratio 1/3, min_rgroup_atoms 2)
  B best-single  every acyclic single bond, keep the cut whose R-GROUP is
                 largest subject to core >= ratio*NHA. Same cut TYPE as A --
                 this isolates how much is the rule versus the SELECTION.
  C multi-bond   cut 2 acyclic bonds at once and take the largest detached
                 component. Reaches middle segments a single cut cannot: a
                 linker between two ring systems detaches whole.
  D brics        RDKit BRICS bonds, which include ring-adjacent cut types A
                 does not consider at all.

Reported as R-group heavy atoms and as a fraction of the parent, since the
complaint is about synthon-likeness rather than absolute size.
"""
import glob, itertools, random, sys
import numpy as np, torch
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "/home/mogan/github/MolPLAtte/molplatte_preprocess/src")
from molplatte_prep.decompose import decompose_molecule, wash

RATIO, MIN_R = 1.0/3.0, 2
KW = dict(method="naveja_recap", ratio=RATIO, include_ring=True,
          max_cores=4, min_rgroup_atoms=MIN_R)


def acyclic_bonds(mol):
    return [b.GetIdx() for b in mol.GetBonds()
            if not b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE
            and b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1]


def pieces(mol, bond_ids):
    """Heavy-atom sizes of the components left after cutting bond_ids."""
    try:
        em = Chem.FragmentOnBonds(mol, list(bond_ids), addDummies=False)
        frags = Chem.GetMolFrags(em, asMols=False)
    except Exception:
        return []
    return [len(f) for f in frags]


def best_single(mol, nha):
    best = 0
    for b in acyclic_bonds(mol):
        sz = pieces(mol, [b])
        if len(sz) != 2:
            continue
        core, rg = max(sz), min(sz)
        if core >= RATIO * nha and rg >= MIN_R:
            best = max(best, rg)
    return best


def multi_bond(mol, nha, k=2):
    bonds = acyclic_bonds(mol)
    best = 0
    for combo in itertools.combinations(bonds, k):
        sz = pieces(mol, combo)
        if len(sz) < 2:
            continue
        core = max(sz)
        if core < RATIO * nha:
            continue
        rest = sorted(sz, reverse=True)[1:]
        if rest and max(rest) >= MIN_R:
            best = max(best, max(rest))
    return best


def brics_best(mol, nha):
    bonds = [b[0] for b in BRICS.FindBRICSBonds(mol)]
    if not bonds:
        return 0
    idx = []
    for i, j in bonds:
        b = mol.GetBondBetweenAtoms(i, j)
        if b is not None:
            idx.append(b.GetIdx())
    best = 0
    for b in idx:
        sz = pieces(mol, [b])
        if len(sz) == 2:
            core, rg = max(sz), min(sz)
            if core >= RATIO * nha and rg >= MIN_R:
                best = max(best, rg)
    return best


files = sorted(glob.glob("/home/mogan/preprocessed/molplatte/"
                         "coconut-flavordb-full/naveja_recap/*/*.pt"))
random.Random(11).shuffle(files)
smis = []
for f in files:
    d = torch.load(f, weights_only=False)
    if d.get("smiles"):
        smis.append(d["smiles"])
    if len(smis) >= int(sys.argv[1] if len(sys.argv) > 1 else 400):
        break

res = {k: [] for k in ("A", "B", "C", "D")}
frac = {k: [] for k in res}
no_decomp = 0
for smi in smis:
    m = wash(smi, remove_stereo=False, neutralise=False)
    if m is None:
        continue
    nha = m.GetNumHeavyAtoms()
    if nha < 6:
        continue
    try:
        _, decs = decompose_molecule(m, do_wash=False, **KW)
    except Exception:
        decs = []
    a = max((len(r.rgroup_atoms) for d in decs for r in d.rgroups), default=0)
    if not decs:
        no_decomp += 1
    for key, val in (("A", a), ("B", best_single(m, nha)),
                     ("C", multi_bond(m, nha)), ("D", brics_best(m, nha))):
        if val:
            res[key].append(val); frac[key].append(val / nha)

print(f"molecules sampled {len(smis)}   no decomposition under A: {no_decomp} "
      f"({100*no_decomp/max(len(smis),1):.1f}%)\n")
name = {"A": "A current (naveja_recap)", "B": "B best single acyclic cut",
        "C": "C multi-bond (2 cuts)", "D": "D BRICS bonds"}
print(f"{'strategy':<28}{'n':>6}{'median':>8}{'mean':>7}{'p90':>6}{'max':>6}"
      f"{'median R/parent':>17}{'>=6 atoms':>11}")
for k in ("A", "B", "C", "D"):
    v = np.array(res[k]); f = np.array(frac[k])
    if not len(v):
        print(f"{name[k]:<28}  none"); continue
    print(f"{name[k]:<28}{len(v):>6}{np.median(v):>8.0f}{v.mean():>7.1f}"
          f"{np.percentile(v,90):>6.0f}{v.max():>6}{np.median(f):>17.2f}"
          f"{100*(v>=6).mean():>10.0f}%")
