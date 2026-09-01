#!/usr/bin/env python
"""Does the assembly head actually produce valid molecules?

The per-attribute recovery accuracies (~99%) say the head predicts joint
chemistry well. They do NOT say a molecule can be rebuilt from those
predictions: 99% per attribute across 7 attributes and k joints compounds, and a
single wrong bond order or H-count can make the result unsanitisable.

MolDAM ships a callback named ``MolecularReassembly``, but it measures
attribute agreement and a Hungarian bijection -- it never builds a molecule or
calls RDKit. This script does the missing check.

Three reassemblies per instance, all through ``attach_rgroups``:

``true``     joint chemistry taken from ``linker_metas`` -- the CEILING. If this
             is not ~100% the reassembly machinery itself is broken, not the model.
``predicted`` joint chemistry from the head's argmax -- the number that matters.
``masked``   no recovery at all, MASK sentinels left in place -- the FLOOR,
             i.e. what retrieval alone gives you.

Attributes the head does not recover (formal_charge and edge_is_aromatic are
structurally constant on this corpus; chiral_tag and bond_dir are excluded as
order-/serialisation-dependent) are held at their true values in the
``predicted`` arm. For the degenerate pair that is not a concession -- the true
value IS the only value.

Validity is RDKit ``SanitizeMol``; ``exact`` additionally requires the rebuilt
canonical SMILES to equal the original molecule's.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
RDLogger.DisableLog("rdApp.*")

from data_modules import MolPalleteDataset, collate_molpallete  # noqa: E402
from data_modules.molpla_prep_bridge import _ensure_importable  # noqa: E402,F401
from nnet_modules import MolPallete  # noqa: E402

from molpallete_prep.graph_ops import attach_rgroups  # noqa: E402
from molpallete_prep.mol_features import pyg_to_mol  # noqa: E402


def _canonical(data) -> str | None:
    """Sanitised canonical SMILES, or None if it is not a real molecule.

    A dummy atom disqualifies the result. ``pyg_to_mol`` maps a MASK
    ``atomic_num`` to a dummy (Z=0), and a molecule containing ``*`` sanitises
    perfectly well -- so a plain SanitizeMol check scores the unrecovered
    "masked" arm at 100% valid, which is meaningless. An unfilled joint is
    exactly the failure this evaluation exists to detect.
    """
    try:
        mol = pyg_to_mol(data, sanitize=False)
        Chem.SanitizeMol(mol)
        if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _predicted_metas(sample, out, rows):
    """Overlay the head's argmax onto a copy of the joint metadata.

    ``rows`` are this sample's row indices into the batch-level joint tensors.
    Attributes the head does not predict keep their true values.
    """
    atom_pred = {a: v.argmax(-1).cpu() for a, v in (out.get("assembly_atom_pred") or {}).items()}
    bond_pred = {a: v.argmax(-1).cpu() for a, v in (out.get("assembly_bond_pred") or {}).items()}
    metas, bond_feats = {}, {}
    for row, lid in zip(rows, sample.joint_linker_ids):
        true = dict(sample.joint_metas.get(lid, {}))
        af = dict(true.get("atom_features", {}))
        bf = dict(true.get("cut_bond_features", {}))
        for a, p in atom_pred.items():
            if a == "chirality_specified":
                continue  # boolean proxy, not a chiral_tag value -- cannot write back
            af[a] = int(p[row])
        for a, p in bond_pred.items():
            bf[a] = int(p[row])
        true["atom_features"] = af
        true["cut_bond_features"] = bf
        metas[lid] = true
        bond_feats[lid] = bf
    return metas, bond_feats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True, help="the run's .hydra/config.yaml")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, default=1000, help="molecules to test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    cfg = OmegaConf.load(args.config)
    model = MolPallete(**OmegaConf.to_container(cfg.nnet_module_kwargs, resolve=True))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if any("assembly" in k for k in missing):
        print("ERROR: this checkpoint has no assembly head -- nothing to evaluate.",
              file=sys.stderr)
        return 1
    model.to(args.device).eval()

    ds = MolPalleteDataset(args.corpus, condvec_dim=cfg.nnet_module_kwargs.condvec_dim,
                           need_assembly_targets=True, seed=0)
    counts = collections.Counter()
    n_joints = collections.Counter()

    done = 0
    for start in range(0, min(args.n, len(ds)), args.batch_size):
        samples = [ds[i] for i in range(start, min(start + args.batch_size, args.n))]
        batch = collate_molpallete(samples)
        dev = {k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.no_grad():
            out = model(dev)

        js = batch["joint_sample"].tolist()
        by_sample = collections.defaultdict(list)
        for row, s in enumerate(js):
            by_sample[s].append(row)

        for si, sample in enumerate(samples):
            rows = by_sample.get(si, [])
            if len(rows) != len(sample.R):
                continue          # a joint was dropped in collate; not reassemblable
            done += 1
            n_joints[len(rows)] += 1
            original = _canonical(sample.G)

            # ceiling: true joint chemistry
            try:
                t = attach_rgroups(sample.P, sample.R, restore_features=True)
                s_true = _canonical(t)
            except Exception:
                s_true = None

            # floor: no recovery, MASK left in place
            try:
                m = attach_rgroups(sample.P, sample.R, restore_features=False)
                s_mask = _canonical(m)
            except Exception:
                s_mask = None

            # the real test: the head's predictions
            try:
                metas, bfeats = _predicted_metas(sample, out, rows)
                P2 = sample.P.clone() if hasattr(sample.P, "clone") else sample.P
                P2.linker_metas = metas
                p = attach_rgroups(P2, sample.R, restore_features=True,
                                   bond_features=bfeats)
                s_pred = _canonical(p)
            except Exception:
                s_pred = None

            for tag, smi in (("true", s_true), ("predicted", s_pred), ("masked", s_mask)):
                if smi is not None:
                    counts[f"{tag}/valid"] += 1
                    if original is not None and smi == original:
                        counts[f"{tag}/exact"] += 1
            # Does the head reproduce the TRUE reassembly, instance by instance?
            # Equal aggregate rates would not establish this.
            if s_true is not None:
                counts["ceiling_reachable"] += 1
                if s_pred == s_true:
                    counts["pred_matches_true"] += 1

    print(f"\nreassembly over {done:,} instances "
          f"(joints per instance: {dict(sorted(n_joints.items()))})\n")
    print(f"  {'joint chemistry':<14s} {'valid':>8s} {'exact match':>13s}")
    for tag in ("true", "predicted", "masked"):
        v = counts[f"{tag}/valid"] / max(done, 1)
        e = counts[f"{tag}/exact"] / max(done, 1)
        print(f"  {tag:<14s} {v:>8.2%} {e:>13.2%}")
    reach = counts["ceiling_reachable"]
    if reach:
        print(f"\n  predicted reproduces the true reassembly on "
              f"{counts['pred_matches_true']}/{reach} = "
              f"{counts['pred_matches_true']/reach:.2%} of instances where the ceiling is reachable")
    print("\n  true      = ceiling (reassembly machinery working)")
    print("  predicted = the assembly head's actual contribution")
    print("  masked    = floor: what retrieval alone leaves you with")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
