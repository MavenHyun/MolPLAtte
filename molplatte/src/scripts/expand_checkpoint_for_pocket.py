#!/usr/bin/env python3
"""Widen a flavor-only checkpoint so a pocket-conditioned model can load it.

STEP 1 pretrains with a flavor-only condition vector, so the query projector
takes ``hidden_dim + 24 = 324`` inputs. STEP 2 adds a projected pocket half and
takes ``hidden_dim + 24 + pocket_dim = 356``. Exactly one tensor changes shape:
``nnet.query_projector.projection.0.weight``.

The new columns are filled with ZEROS, and that choice is what makes the
transfer exact rather than approximate. ``PocketConditioning`` zero-initialises
its own output layer, so on the first step of STEP 2 the pocket half is zero
going in AND zero-weighted coming out. The model therefore computes bit-identical
outputs to the STEP 1 checkpoint and has to earn every subsequent departure from
it, instead of starting from a random perturbation of a converged solution.

The alternative -- storing 1280 zero floats per record so STEP 1's corpus is
already 1304 wide -- would cost roughly 5 GB of zeros across the pretraining
corpus to avoid a matrix reshape.

Verified rather than asserted: --check reloads the result and confirms the old
columns survived untouched and the new ones are exactly zero.
"""
from __future__ import annotations

import argparse
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
    new = w.new_zeros((out_dim, target_in))
    new[:, :old_in] = w
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
        assert w[:, old_in:].abs().sum().item() == 0.0, "new columns are not zero"
        print("  verified: old columns identical, new columns exactly zero")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
