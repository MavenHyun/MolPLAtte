#!/usr/bin/env python3
"""Fit the fixed PCA basis that PocketConditioning projects pockets through.

One basis PER FOLD, fitted on that fold's TRAINING records only. Fitting once
over all pockets would leak: PCA directions are chosen to explain the variance
of the data they see, so a global basis is partly a description of the held-out
receptors, and the fold would no longer be measuring generalisation.

The basis is orthonormal, so the projection is an isometry on the retained
subspace and the metric structure the 2026-09-09 probe found (Tanimoto 0.35 vs
0.12 to a random pocket, across unseen receptors) survives the reduction. Per
-component scaling to unit variance is stored alongside, so the 32 pocket
features and the 24 flavour bits reach the query projector on comparable scales.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch


def load(corpus: Path):
    rows = []
    for f in sorted(glob.glob(str(corpus / "**" / "*.pt"), recursive=True)):
        d = torch.load(f, weights_only=False)
        m = d.get("meta") or {}
        pe = m.get("pocket_embedding")
        if pe is None:
            continue
        rows.append((int(m.get("fold", -1)), np.asarray(pe, dtype=np.float64)))
    return rows


def fit(X: np.ndarray, k: int):
    mean = X.mean(axis=0)
    Xc = X - mean
    # SVD rather than a covariance eigendecomposition: 1280 dims from ~200
    # samples makes the covariance rank-deficient and its eigenvectors unstable.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vt.shape[0])
    comp = Vt[:k]
    proj = Xc @ comp.T
    scale = proj.std(axis=0)
    evr = float((S[:k] ** 2).sum() / (S**2).sum())
    return comp, mean, scale, evr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path,
                    default=Path("/home/mogan/preprocessed/molplatte/"
                                 "tastepocket_corpus/naveja_recap"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/home/mogan/preprocessed/molplatte/pocket_basis"))
    ap.add_argument("--dim", type=int, default=32)
    args = ap.parse_args()

    rows = load(args.corpus)
    if not rows:
        raise SystemExit(f"no records with a pocket_embedding under {args.corpus}")
    folds = sorted({f for f, _ in rows})
    X_all = np.stack([x for _, x in rows])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} pocket records, folds {folds}, dim {X_all.shape[1]} -> {args.dim}\n")

    report = {}
    for k in folds:
        Xtr = np.stack([x for f, x in rows if f != k])
        comp, mean, scale, evr = fit(Xtr, args.dim)
        out = args.out_dir / f"fold{k}.npz"
        np.savez(out, components=comp, mean=mean, scale=scale)
        report[f"fold{k}"] = dict(n_train=len(Xtr), evr=evr)
        print(f"  fold {k}: fitted on {len(Xtr):>4} training records, "
              f"variance retained {100*evr:.1f}%  -> {out.name}")

    comp, mean, scale, evr = fit(X_all, args.dim)
    np.savez(args.out_dir / "full.npz", components=comp, mean=mean, scale=scale)
    report["full"] = dict(n_train=len(X_all), evr=evr)
    print(f"  full   : fitted on {len(X_all):>4} records, "
          f"variance retained {100*evr:.1f}%  -> full.npz")
    print("\n  full.npz is for the FINAL model only, which trains on everything;")
    print("  never score a fold with it.")

    # Orthonormality is the property the isometry argument rests on, so assert it.
    err = float(np.abs(comp @ comp.T - np.eye(comp.shape[0])).max())
    if err > 1e-8:
        raise SystemExit(f"basis is not orthonormal (max deviation {err:.2e})")
    print(f"  orthonormality checked: max |CC^T - I| = {err:.2e}")

    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
