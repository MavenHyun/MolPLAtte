#!/usr/bin/env python3
"""Mean-pool ESM-2 residue embeddings over the geometric pocket.

The pocket is selected in 3D (residues within 10 A of the ligand) and only then
pooled from a 1D language model's per-residue output. The embedding is 1D; the
*selection* is not, which is what keeps this usable for GPCRs where pocket
residues are far apart in sequence.

Pooling is a mean over every pocket residue across every chain the pocket
touches -- 1,392 of 2,269 tastepocket sites span more than one chain. Pooling
per chain and then averaging the chains would give a 5-residue contact the same
weight as a 40-residue wall.

THE BOS OFFSET IS THE TRAP HERE. ESM-2's tokenizer prepends <cls>, so residue
``i`` of the sequence is token ``i + 1`` of the output. Forgetting it shifts
every pocket by one residue and raises nothing: the vectors stay finite, the
shapes stay right, and retrieval just quietly gets worse. ``--verify-offset``
asserts the alignment against the tokenizer instead of trusting this comment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

MODEL = "facebook/esm2_t33_650M_UR50D"

#: A "chain" this short is a peptide fragment or a modelling artifact, not a
#: protein; ESM has no context to embed it and it would add noise to the mean.
MIN_CHAIN_LENGTH = 20


def verify_offset(tokenizer) -> int:
    """Return the number of prefix tokens, checked rather than assumed."""
    probe = "ACDEFGHIKLMNPQRSTVWY"
    ids = tokenizer(probe, return_tensors="pt")["input_ids"][0]
    n_prefix = 0
    while n_prefix < len(ids) and ids[n_prefix] in tokenizer.all_special_ids:
        n_prefix += 1
    decoded = tokenizer.convert_ids_to_tokens(ids[n_prefix : n_prefix + len(probe)])
    if "".join(decoded) != probe:
        raise RuntimeError(
            f"tokenizer does not align at offset {n_prefix}: "
            f"got {''.join(decoded)!r}, expected {probe!r}"
        )
    if len(ids) != len(probe) + n_prefix + 1:
        raise RuntimeError(
            f"unexpected token count {len(ids)} for a {len(probe)}-residue probe"
        )
    return n_prefix


@torch.no_grad()
def embed_sequences(seqs: List[str], model, tokenizer, device, batch_size: int,
                    n_prefix: int) -> Dict[str, np.ndarray]:
    """sequence -> ``[L, D]`` per-residue embeddings, BOS/EOS already stripped."""
    out: Dict[str, np.ndarray] = {}
    order = sorted(seqs, key=len)          # length-sorted batches = less padding
    for i in range(0, len(order), batch_size):
        chunk = order[i : i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state.float().cpu().numpy()
        for j, seq in enumerate(chunk):
            out[seq] = hidden[j, n_prefix : n_prefix + len(seq)]
        print(f"  embedded {min(i + batch_size, len(order))}/{len(order)}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pockets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="cuda:0",
                    help="index within CUDA_VISIBLE_DEVICES, not a global GPU id")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    records = [json.loads(l) for l in args.pockets.open()]
    seqs = sorted({d["sequence"] for r in records for d in r["chains"].values()
                   if len(d["sequence"]) >= MIN_CHAIN_LENGTH})
    print(f"records {len(records)}   unique chain sequences {len(seqs)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    n_prefix = verify_offset(tokenizer)
    print(f"tokenizer prefix tokens: {n_prefix} (verified against a 20-mer probe)")

    model = AutoModel.from_pretrained(args.model).to(device).eval()
    missing = [n for n, _ in model.named_parameters() if "pooler" in n]
    if missing:
        print(f"note: {len(missing)} pooler params are randomly initialised and unused "
              "(we read last_hidden_state, not the pooled output)")

    print(f"embedding on {device} ...")
    per_res = embed_sequences(seqs, model, tokenizer, device, args.batch_size, n_prefix)
    dim = next(iter(per_res.values())).shape[1]
    print(f"embedding dim {dim}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    vecs: List[np.ndarray] = []
    skipped = 0
    for r in records:
        rows = []
        for d in r["chains"].values():
            seq = d["sequence"]
            emb = per_res.get(seq)
            if emb is None:
                continue
            idx = [i for i in d["pocket_index"] if 0 <= i < len(emb)]
            if idx:
                rows.append(emb[idx])
        if not rows:
            skipped += 1
            continue
        # One mean over every pocket residue of every chain, so a chain
        # contributing 40 residues outweighs one contributing 5.
        keys.append(f"{r['pdb_id']}_{r['ccd']}_{r['instance']}")
        vecs.append(np.concatenate(rows, axis=0).mean(axis=0))

    arr = np.stack(vecs).astype(np.float32)
    np.savez_compressed(args.out, keys=np.array(keys), embeddings=arr)
    print(f"\npocket embeddings {arr.shape}   skipped {skipped}")
    print(f"norm: mean {np.linalg.norm(arr, axis=1).mean():.2f} "
          f"min {np.linalg.norm(arr, axis=1).min():.2f} "
          f"max {np.linalg.norm(arr, axis=1).max():.2f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
