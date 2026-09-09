#!/usr/bin/env python3
"""How reliable is the assembly head?

Retrieval says WHICH R-group belongs at a joint. Assembly says HOW to attach it:
the joint is masked, so the shared linker atom's identity and the bond that
reforms were replaced by MASK sentinels, and without predicting them back a
retrieved R-group cannot be bonded on at all.

Reliability is measured three ways, because they fail differently:

1. **Per-attribute accuracy.** Each of atomic_num / formal_charge /
   total_num_hs / bond_type is an independent classifier. Reporting only a
   combined number hides which one is weak, and they are not equally hard --
   bond_type has 22 classes but is nearly always SINGLE, while atomic_num has
   128 and actually varies.

2. **Round-trip exactness.** Detach an R-group, reattach THE SAME one using
   predicted chemistry, and compare canonical SMILES against the parent. This
   is the only end-to-end check: it is insensitive to which attribute was wrong
   and answers "would a user get the right molecule".

3. **Aromaticity preservation.** RDKit sanitises a molecule whose aromatic ring
   has been broken by a wrong hydrogen count, so "it parsed" and "it is
   chemically right" are different claims and a combined validity number
   conflates them.

A ground-truth control runs alongside: the same path with the STORED joint
chemistry instead of predictions. It isolates the plumbing from the head, so a
low score can be attributed to one or the other rather than guessed at.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdkit import Chem, RDLogger  # noqa: E402

from lead_optimization import DECOMP_KWARGS, LeadOptimizer  # noqa: E402
from molplatte_prep.decompose import decompose_molecule, wash  # noqa: E402
from molplatte_prep.graph_ops import attach_rgroups  # noqa: E402
from molplatte_prep.mol_features import mol_to_pyg, pyg_to_mol  # noqa: E402
from molplatte_prep.molpla_instance import build_instance  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def n_aromatic(smiles: str) -> int:
    m = Chem.MolFromSmiles(smiles) if smiles else None
    return sum(1 for a in m.GetAtoms() if a.GetIsAromatic()) if m else -1


def corpus_smiles(corpus: Path, n: int, seed: int):
    files = sorted(corpus.rglob("*.pt"))
    random.Random(seed).shuffle(files)
    out = []
    for f in files:
        if len(out) >= n:
            break
        try:
            d = torch.load(f, weights_only=False)
        except Exception:  # noqa: BLE001
            continue
        if d.get("smiles"):
            out.append(d["smiles"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True,
                    help="corpus to draw evaluation molecules from")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--condvec-dim", type=int, default=0)
    ap.add_argument("--pocket-input-dim", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args()

    opt = LeadOptimizer.load(args.checkpoint, args.vocab, device=args.device,
                             condvec_dim=args.condvec_dim,
                             pocket_input_dim=args.pocket_input_dim,
                             assembly_head="AssemblyHead")
    if not opt.has_assembly_head:
        print("checkpoint has no TRAINED assembly head; nothing to evaluate")
        return 1

    smiles = corpus_smiles(args.corpus, args.n, args.seed)
    print(f"drew {len(smiles)} molecules from {args.corpus.parent.name}\n")

    head = opt.model.nnet["assembly_head"]
    attr_ok: Counter = Counter()
    attr_n: Counter = Counter()
    rt_pred = rt_truth = arom_ok = considered = 0
    fails: Counter = Counter()

    for smi in smiles:
        mol = wash(smi, remove_stereo=False, neutralise=False)
        if mol is None:
            continue
        try:
            data = mol_to_pyg(mol)
            _, decs = decompose_molecule(mol, do_wash=False, **DECOMP_KWARGS)
        except Exception:  # noqa: BLE001
            continue
        if not decs:
            continue
        dec = decs[0]
        k = len(dec.rgroups)
        try:
            inst = build_instance(data, dec, [i != 0 for i in range(k)],
                                  mol_id="eval", store_orig=True,
                                  compute_hashes=False)
        except Exception:  # noqa: BLE001
            continue
        lid = int(inst.joint_linker_ids[0])
        R = inst.R[0] if isinstance(inst.R, (list, tuple)) else inst.R
        parent = Chem.MolToSmiles(mol)
        considered += 1

        # --- per-attribute: predicted vs the stored truth at this joint
        meta = (getattr(inst.P, "linker_metas", {}) or {}).get(lid) or {}
        truth_atom = meta.get("atom_features") or {}
        truth_bond = meta.get("cut_bond_features") or {}
        try:
            from torch_geometric.data import Batch

            import copy
            tpl = copy.deepcopy(inst.P)
            rg = copy.deepcopy(R)
            core_at = int((tpl.linker_id == lid).nonzero()[0].item())
            rg_at = int(rg.is_linker.nonzero()[0].item())
            batch = Batch.from_data_list([tpl, rg]).to(opt.device)
            with torch.no_grad():
                H = opt.model.nnet["graph_encoder"](batch).node_embeddings
                off = int((batch.batch == 0).sum().item())
                fused = head._fuse(H[core_at].unsqueeze(0),
                                   H[off + rg_at].unsqueeze(0))
                pred = {a: int(h(fused).argmax(-1).item())
                        for a, h in head.node_heads.items()}
                pred.update({a: int(h(fused).argmax(-1).item())
                             for a, h in head.edge_heads.items()})
        except Exception:  # noqa: BLE001
            fails["predict"] += 1
            continue
        for a, v in pred.items():
            t = truth_atom.get(a, truth_bond.get(a))
            if t is None:
                continue
            attr_n[a] += 1
            attr_ok[a] += int(v == t)

        # --- round trip with PREDICTED chemistry
        got, err = opt.assemble(inst.P, R, lid)
        if got is None:
            fails[err.split(":")[0][:28] or "assemble"] += 1
        else:
            rt_pred += int(got == parent)
            arom_ok += int(n_aromatic(got) >= n_aromatic(parent))

        # --- control: the same path with STORED chemistry
        try:
            merged = attach_rgroups(inst.P, [R], linker_ids=[lid],
                                    restore_features=True)
            rt_truth += int(Chem.MolToSmiles(pyg_to_mol(merged, sanitize=True))
                            == parent)
        except Exception:  # noqa: BLE001
            pass

    if not considered:
        print("no molecules decomposed; nothing to report")
        return 1

    print(f"evaluated {considered} joints\n")
    print("1. PER-ATTRIBUTE ACCURACY (independent classifiers)")
    for a in sorted(attr_n, key=lambda x: -attr_n[x]):
        n = attr_n[a]
        print(f"   {a:16s} {100*attr_ok[a]/n:6.1f}%   ({attr_ok[a]}/{n})")

    print("\n2. ROUND TRIP -- reattach the ORIGINAL R-group, compare to parent")
    print(f"   with PREDICTED chemistry  {100*rt_pred/considered:6.1f}%   "
          f"({rt_pred}/{considered})")
    print(f"   with STORED chemistry     {100*rt_truth/considered:6.1f}%   "
          f"({rt_truth}/{considered})   <- plumbing control")
    gap = rt_truth - rt_pred
    print(f"   gap attributable to the head: {gap} of {considered} "
          f"({100*gap/considered:.1f} points)")

    print("\n3. AROMATICITY")
    print(f"   preserved                 {100*arom_ok/considered:6.1f}%   "
          f"({arom_ok}/{considered})")
    print("   RDKit sanitises a broken aromatic ring without complaint, so this")
    print("   is reported apart from whether the SMILES parsed.")

    if fails:
        print("\n   failures:")
        for k, v in fails.most_common():
            print(f"     {k:30s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
