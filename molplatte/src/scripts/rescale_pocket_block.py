#!/usr/bin/env python3
"""Scale the query projector's pocket columns to parity with its flavour columns.

WHY THIS IS SAFE. The query projector takes [hidden | flavour | pocket]. The
pocket slice is produced by PocketConditioning, whose output layer is
ZERO-initialised -- so at initialisation the pocket input is exactly zero and
these columns multiply zero. Rescaling them therefore cannot change a single
prediction at init. Verified by scoring: the rescaled checkpoint must reproduce
the original's retrieval numbers exactly, and the caller is expected to check.

WHY IT MATTERS ANYWAY. The gradient reaching the zero-init adapter is
proportional to these same columns. Measured on exp-wide512-pocket.pt they sit
at untrained PyTorch default init (mean|w| 0.0209, i.e. 0.5/sqrt(568)) while the
flavour columns trained up to 0.8281 -- 39.6x larger. The two smallnesses
compound: weak columns give a weak gradient into the adapter, the adapter stays
near zero (max|w| ~0.001 after 40 epochs), and the pocket never reaches the
query. That confounds "pockets carry no information" with "the pocket block was
initialised too weakly to find out".

This equalises the two conditioning blocks so the ladder can be re-run with the
pocket given the same footing as flavour.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

KEY = "query_projector.projection.0.weight"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--flavour", type=int, default=24)
    ap.add_argument("--scale", type=float, default=None,
                    help="explicit factor; default = flavour/pocket mean|w| parity")
    a = ap.parse_args()

    obj = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    keys = [k for k in sd if k.endswith(KEY)]
    if len(keys) != 1:
        raise SystemExit(f"expected exactly one {KEY}, found {len(keys)}")
    k = keys[0]
    W = sd[k]
    H, F = a.hidden, a.flavour
    if W.shape[1] <= H + F:
        raise SystemExit(f"{k} has width {W.shape[1]}, no pocket block beyond {H+F}")

    pocket = W[:, H + F:]
    flavour = W[:, H:H + F]
    before = float(pocket.abs().mean())
    fmean = float(flavour.abs().mean())
    scale = a.scale if a.scale is not None else fmean / max(before, 1e-12)

    W[:, H + F:] = pocket * scale
    after = float(W[:, H + F:].abs().mean())

    print(f"  {k}  shape={tuple(W.shape)}  pocket cols {H+F}:{W.shape[1]}")
    print(f"  flavour mean|w| {fmean:.6f}")
    print(f"  pocket  mean|w| {before:.6f} -> {after:.6f}   (x{scale:.2f})")
    ratio = after / fmean
    assert 0.9 < ratio < 1.1 or a.scale is not None, f"parity not reached: {ratio:.3f}"

    sd[k] = W
    if isinstance(obj, dict) and "state_dict" in obj:
        obj["state_dict"] = sd
    else:
        obj = sd
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, a.out)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
