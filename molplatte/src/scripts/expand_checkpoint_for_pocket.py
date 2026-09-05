#!/usr/bin/env python3
"""Widen a flavor-only checkpoint so a pocket-conditioned model can load it.

STEP 1 pretrains with a flavor-only condition vector, so the query projector
takes ``hidden_dim + 24 = 324`` inputs. STEP 2 adds a projected pocket half and
takes ``hidden_dim + 24 + pocket_dim = 356``. Exactly one tensor changes shape:
``nnet.query_projector.projection.0.weight``.

The new columns are RANDOM, and that is load-bearing. Zeroing them as well is
the obvious thing to do and it silently disables the pocket path forever:

    d(loss)/d(pocket)   = W_pocket^T . d(loss)/d(z)   = 0  because W_pocket = 0
    d(loss)/d(W_pocket) = d(loss)/d(z) (x) pocket     = 0  because pocket   = 0

Each zero keeps the other pinned, so both stay at exactly zero for the whole
run. Measured: with zero columns the gradient reaching PocketConditioning is
0.000000; with random columns it is 437.4. The first STEP 2 sweep was run with
zero columns and finished with |output layer| = 0.0000 in all five folds -- a
flavour-only finetune wearing a pocket-conditioned label, and every metric
looked normal.

The transfer stays exact anyway, because only ONE side needs to be zero:
``PocketConditioning`` zero-initialises its own output layer, so the projected
pocket is the zero vector on step 0 and contributes nothing no matter what the
query columns hold. The model computes bit-identical outputs to the STEP 1
checkpoint and must earn any departure -- but now it CAN.

The alternative -- storing 1280 zero floats per record so STEP 1's corpus is
already 1304 wide -- would cost roughly 5 GB of zeros across the pretraining
corpus to avoid a matrix reshape.

Verified rather than asserted: --check reloads the result and confirms the old
columns survived untouched and the new ones are NOT zero.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

QUERY_W = "nnet.query_projector.projection.0.weight"


def expand(state: dict, target_in: int, key: str = QUERY_W) -> tuple[dict, int, int]:
    if key not in state:
        raise KeyError(
            f"{key} not in the checkpoint; keys look like "
            f"{[k for k in list(state)[:3]]}"
        )
    w = state[key]
    out_dim, old_in = w.shape
    if old_in == target_in:
        return state, old_in, target_in
    if old_in > target_in:
        raise ValueError(
            f"checkpoint query projector takes {old_in} inputs, which is WIDER "
            f"than the requested {target_in}. Narrowing would discard trained "
            "weights; this script only widens."
        )
    new = w.new_empty((out_dim, target_in))
    new[:, :old_in] = w
    # nn.Linear's own init for this fan-in. NOT zeros -- see the module
    # docstring; zeros here deadlock the pocket path permanently.
    bound = 1.0 / math.sqrt(target_in)
    new[:, old_in:].uniform_(-bound, bound)
    state = dict(state)
    state[key] = new
    return state, old_in, target_in


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden-dim", type=int, default=300)
    ap.add_argument("--flavor-dim", type=int, default=24)
    ap.add_argument("--pocket-dim", type=int, default=32)
    ap.add_argument("--check", action="store_true", default=True)
    args = ap.parse_args()

    target = args.hidden_dim + args.flavor_dim + args.pocket_dim
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    original = state[QUERY_W].clone()

    state, old_in, new_in = expand(state, target)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)

    print(f"query projector input {old_in} -> {new_in}")
    print(f"  hidden {args.hidden_dim} + flavor {args.flavor_dim} "
          f"+ pocket {args.pocket_dim}")
    print(f"  tensors carried over: {len(state)}")

    if args.check:
        got = torch.load(args.out, map_location="cpu", weights_only=False)
        w = got[QUERY_W]
        assert w.shape[1] == target, f"reloaded width {w.shape[1]} != {target}"
        assert torch.equal(w[:, :old_in], original), "existing columns changed"
        assert w[:, old_in:].abs().sum().item() > 0.0, (
            "new columns are zero -- this deadlocks the pocket path")
        print("  verified: old columns identical, new columns non-zero "
              f"(|w| {w[:, old_in:].abs().mean().item():.5f})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
