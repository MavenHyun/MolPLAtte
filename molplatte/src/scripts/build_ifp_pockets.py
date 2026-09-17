"""Typed non-covalent interaction descriptors for the tastepocket pockets.

Builds FOUR representations. The split between them is the whole point:

  apo-shell   residue type x distance shell.  Reproduces the published `shell`
              descriptor (+0.0325 cross-family).  Here as a PIPELINE CONTROL --
              if this does not land near the published number, the geometry or
              the residue selection is wrong and nothing else can be trusted.

  apo-pharm   pharmacophore CLASS (donor/acceptor/hydrophobic/aromatic/
              cationic/anionic) x distance shell.  What the pocket PRESENTS.
              Uses the ligand's POSITION -- the pocket is already defined by a
              10 A cutoff around it, so this is information the existing
              baseline also consumed -- but never the ligand's CHEMISTRY.
              This is the deployable one: computable for any receptor you want
              to design against, with no ligand in hand.

  holo-ifp    ProLIF typed interactions with the crystal ligand, as
              residue type x interaction type.
  holo-itype  the same, collapsed to interaction-type totals.

              BOTH holo variants are ORACLES.  The probe task is "predict the
              ligand"; these descriptors are computed FROM that ligand, so they
              are circular by construction and CANNOT be deployed.  They are
              here as a CEILING: if typed interactions cannot beat ESM-2 even
              with the answer in hand, the representation is dead and no amount
              of apo engineering rescues it.  Reporting holo as deployable
              would repeat the condition-vector leak (commit 1a6c709).
"""
import json, sys, warnings, tempfile, subprocess, os
from pathlib import Path
from collections import Counter
import numpy as np

warnings.filterwarnings("ignore")
import prody; prody.confProDy(verbosity="none")
from rdkit import Chem, RDLogger; RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem

D = Path("/home/mogan/preprocessed/molplatte/tastepocket")
STRUCT = Path("/home/mogan/datasets/tastepocket/structures/cif")

SHELLS = [0.0, 4.0, 6.0, 8.0, 10.0]          # distance bins, angstrom
RESTYPES = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
            "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
RIDX = {r: i for i, r in enumerate(RESTYPES)}

# side-chain pharmacophore capability; a residue may be in several classes
PHARM = {
    "donor":       {"ARG","LYS","TRP","ASN","GLN","HIS","SER","THR","TYR","CYS"},
    "acceptor":    {"ASP","GLU","ASN","GLN","HIS","SER","THR","TYR"},
    "hydrophobic": {"ALA","VAL","LEU","ILE","MET","PHE","PRO","TRP","CYS"},
    "aromatic":    {"PHE","TYR","TRP","HIS"},
    "cationic":    {"ARG","LYS","HIS"},
    "anionic":     {"ASP","GLU"},
}
PCLASSES = list(PHARM)

ITYPES = ["Hydrophobic","HBDonor","HBAcceptor","PiStacking","EdgeToFace",
          "FaceToFace","Anionic","Cationic","CationPi","PiCation",
          "XBDonor","XBAcceptor","MetalDonor","MetalAcceptor","VdWContact"]
IIDX = {t: i for i, t in enumerate(ITYPES)}


def shell_of(d):
    for i in range(len(SHELLS) - 1):
        if SHELLS[i] <= d < SHELLS[i + 1]:
            return i
    return None


def load_structure(pdb_id):
    for ext in (".cif", ".pdb"):
        p = STRUCT / f"{pdb_id}{ext}"
        if p.exists():
            return prody.parseMMCIF(str(p)) if ext == ".cif" else prody.parsePDB(str(p))
    return None


def apo_vectors(structure, rec):
    """(shell, pharm) descriptors -- ligand position only, no ligand chemistry."""
    chain, resnum = rec["instance"].split("_")
    sel = structure.select(f"chain {chain} and resname {rec['ccd']} and resnum {int(resnum)}")
    if sel is None:
        return None
    centroid = sel.getCoords().mean(0)

    shell = np.zeros((len(RESTYPES), len(SHELLS) - 1), dtype=np.float32)
    pharm = np.zeros((len(PCLASSES), len(SHELLS) - 1), dtype=np.float32)
    for ch, num, rname in rec["residues"]:
        r = structure.select(f"chain {ch} and resnum {int(num)} and protein")
        if r is None:
            continue
        d = np.linalg.norm(r.getCoords() - centroid, axis=1).min()
        s = shell_of(d)
        if s is None or rname not in RIDX:
            continue
        shell[RIDX[rname], s] += 1
        for ci, cls in enumerate(PCLASSES):
            if rname in PHARM[cls]:
                pharm[ci, s] += 1
    return shell.ravel(), pharm.ravel()



# --- structure prep -------------------------------------------------------
# Two defects in the naive RDKit path, both of which fail SILENTLY (the run
# completes and writes a descriptor that simply has no polar interactions in it):
#
#  1. Chem.AddHs creates hydrogens with no PDBResidueInfo. ProLIF splits the
#     protein into residues by that info and DROPS atoms it cannot place, which
#     removes every explicit H -- so the donor SMARTS ("...-[H]") matches
#     nothing and HBDonor/HBAcceptor are identically zero.
#  2. RDKit's PDB reader leaves Asp/Glu/Arg/Lys neutral, so Anionic/Cationic
#     can never fire regardless of geometry.
CHARGE_AT_PH7 = {("ASP", "OD2"): -1, ("GLU", "OE2"): -1,
                 ("ARG", "NH2"): 1,  ("LYS", "NZ"): 1}


def inherit_pdb_info(mol):
    """Give added H the residue info of their parent heavy atom."""
    for a in mol.GetAtoms():
        if a.GetPDBResidueInfo() is not None or not a.GetNeighbors():
            continue
        src = a.GetNeighbors()[0].GetPDBResidueInfo()
        if src is None:
            continue
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName(src.GetResidueName())
        info.SetResidueNumber(src.GetResidueNumber())
        info.SetChainId(src.GetChainId())
        info.SetInsertionCode(src.GetInsertionCode())
        info.SetIsHeteroAtom(src.GetIsHeteroAtom())
        info.SetOccupancy(1.0)
        info.SetTempFactor(0.0)
        info.SetName(f"H{a.GetIdx() % 1000}".ljust(4)[:4])
        a.SetMonomerInfo(info)
    return mol


def set_charges(mol):
    """Protonation state at pH 7 for the ionisable side chains."""
    for a in mol.GetAtoms():
        i = a.GetPDBResidueInfo()
        if i is None:
            continue
        q = CHARGE_AT_PH7.get((i.GetResidueName().strip(), i.GetName().strip()))
        if q is not None:
            a.SetFormalCharge(q)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
    return mol


def prepare_protein(pdb_path):
    pm = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=True)
    if pm is None:
        return None
    pm = set_charges(pm)
    try:
        Chem.SanitizeMol(pm)
    except Exception:
        return None
    return inherit_pdb_info(Chem.AddHs(pm, addCoords=True))



# Corpus SMILES are neutral forms. At pH 7 a carboxylate is deprotonated and an
# aliphatic amine is protonated, and taste ligands are full of both -- leaving
# them neutral makes Anionic/Cationic unfirable no matter the geometry.
_ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
_AMINE = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(N-[!#6]);!$(N~[#6]=[O,N,S]);!$(N-a);!$([N+])]")


def ligand_ph7(mol):
    """Deprotonate carboxylic acids, protonate aliphatic amines."""
    for match in mol.GetSubstructMatches(_ACID):
        o = mol.GetAtomWithIdx(match[2])
        o.SetFormalCharge(-1)
        o.SetNoImplicit(True)
        o.SetNumExplicitHs(0)
    for (n,) in mol.GetSubstructMatches(_AMINE):
        a = mol.GetAtomWithIdx(n)
        if a.GetTotalValence() < 4:
            a.SetFormalCharge(1)
            a.SetNumExplicitHs(a.GetTotalNumHs() + 1)
            a.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def holo_vectors(structure, rec, smiles, tmpdir):
    """ProLIF typed interactions with the crystal ligand. ORACLE -- see module docstring.

    Protein goes through RDKit rather than an external protonator: obabel's
    output renumbers residues so ProLIF's neighbour search returns ResidueIds
    that are absent from its own residue table (KeyError mid-run).  n_jobs=1
    because ProLIF otherwise spawns a process pool per complex, which costs
    ~10x the actual work.
    """
    import prolif as plf

    chain, resnum = rec["instance"].split("_")
    # "not hydrogen" at SELECTION time: RDKit silently ignores removeHs=True when
    # sanitize=False, so crystallographic H would survive and be counted against a
    # heavy-atom-only SMILES template, failing the match. Both molecules are
    # re-protonated uniformly below.
    lig_sel = structure.select(
        f"chain {chain} and resname {rec['ccd']} and resnum {int(resnum)} and not hydrogen")
    if lig_sel is None:
        return None, "no_lig_sel"
    sub = " or ".join(f"(chain {c} and resnum {int(n)})" for c, n, _ in rec["residues"])
    prot_sel = structure.select(f"protein and not hydrogen and ({sub})")
    if prot_sel is None:
        return None, "no_prot_sel"

    lp, pp = f"{tmpdir}/lig.pdb", f"{tmpdir}/prot.pdb"
    prody.writePDB(lp, lig_sel)
    prody.writePDB(pp, prot_sel)

    # ligand: crystal coords, bond orders from the corpus SMILES, then explicit H
    lig = Chem.MolFromPDBFile(lp, removeHs=False, sanitize=False)
    if lig is None:
        return None, "lig_unreadable"
    degraded = False
    if smiles:
        tmpl = Chem.MolFromSmiles(smiles)
        if tmpl is not None:
            try:
                lig = AllChem.AssignBondOrdersFromTemplate(tmpl, lig)
            except Exception:
                # The deposited ligand is incomplete (disordered tails unmodelled),
                # so no template can match. Keep the record with PDB-perceived
                # bonds and COUNT it -- silently dropping ~18% of pockets, all of
                # them one lipid, would bias the probe without saying so.
                degraded = True
    try:
        Chem.SanitizeMol(lig)
        lig = ligand_ph7(lig)
        if lig is None:
            return None, "lig_ph7_fail"
        lig = inherit_pdb_info(Chem.AddHs(lig, addCoords=True))
    except Exception:
        return None, "lig_sanitize_fail"

    try:
        pm = prepare_protein(pp)
        if pm is None:
            return None, "prot_unreadable"
        prot = plf.Molecule.from_rdkit(pm)
        fp = plf.Fingerprint(count=True)
        fp.run_from_iterable([plf.Molecule.from_rdkit(lig)], prot,
                             progress=False, n_jobs=1)
        df = fp.to_dataframe()
    except Exception as e:
        return None, f"prolif:{type(e).__name__}"

    mat = np.zeros((len(RESTYPES), len(ITYPES)), dtype=np.float32)
    tot = np.zeros(len(ITYPES), dtype=np.float32)
    for col in df.columns:
        presid, itype = col[1], col[2]
        n = float(df[col].iloc[0])
        if n == 0 or itype not in IIDX:
            continue
        rname = str(presid)[:3].upper()
        tot[IIDX[itype]] += n
        if rname in RIDX:
            mat[RIDX[rname], IIDX[itype]] += n
    return (mat.ravel(), tot), ("degraded_bonds" if degraded else "ok")


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    recs = [json.loads(l) for l in open(D / "pockets.jsonl")]
    if limit:
        recs = recs[:limit]
    ligs = {j["ccd"]: j.get("smiles") for j in map(json.loads, open(D / "ligands.jsonl"))}

    out = {"keys": [], "apo_shell": [], "apo_pharm": [], "holo_ifp": [], "holo_itype": []}
    cache, fails = {}, Counter()
    degraded_keys = []
    itype_hist = Counter()

    with tempfile.TemporaryDirectory() as td:
        for n, rec in enumerate(recs):
            if n % 50 == 0:
                print(f"  {n}/{len(recs)}  ok={len(out['keys'])}  {dict(fails)}", flush=True)
            pid = rec["pdb_id"]
            if pid not in cache:
                cache[pid] = load_structure(pid)
                if len(cache) > 12:
                    cache.pop(next(iter(cache)))
            st = cache[pid]
            if st is None:
                fails["no_structure"] += 1; continue
            a = apo_vectors(st, rec)
            if a is None:
                fails["no_apo"] += 1; continue
            h, why = holo_vectors(st, rec, ligs.get(rec["ccd"]), td)
            if h is None:
                fails[why] += 1; continue
            if why == "degraded_bonds":
                degraded_keys.append(f"{pid}_{rec['ccd']}_{rec['instance']}")
            for t, c in zip(ITYPES, h[1]):
                if c: itype_hist[t] += 1
            out["keys"].append(f"{pid}_{rec['ccd']}_{rec['instance']}")
            out["apo_shell"].append(a[0]); out["apo_pharm"].append(a[1])
            out["holo_ifp"].append(h[0]);  out["holo_itype"].append(h[1])

    print(f"\nbuilt {len(out['keys'])}/{len(recs)}   failures {dict(fails)}")
    print(f"degraded (PDB-perceived bonds, incomplete crystal ligand): {len(degraded_keys)}")
    print("interaction types actually detected (n pockets with >=1):")
    for t in ITYPES:
        print(f"  {t:<16}{itype_hist[t]:>6}")
    np.savez_compressed("/tmp/ifp_pockets.npz",
                        keys=np.array(out["keys"]),
                        degraded=np.array(degraded_keys),
                        **{k: np.stack(v) for k, v in out.items() if k != "keys"})
    print("-> /tmp/ifp_pockets.npz")


if __name__ == "__main__":
    sys.exit(main())
