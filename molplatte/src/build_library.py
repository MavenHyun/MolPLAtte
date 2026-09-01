#!/usr/bin/env python
"""Export a trained R-group library: FAISS index + row metadata.

The library is what makes lead optimization queryable.  Given a core template with
a masked linker and a desired condition vector, you project the query and search
this index for R-groups that fit — MolPLA's R-Group Retrieval task, run as
inference rather than as a validation metric.

Two halves, built at different times:

* the **vocabulary** (static, per corpus) — built by
  ``molplatte_preprocess/enumerate_rgroups.py``;
* the **index** (this script) — the vocabulary's graphs embedded with a specific
  checkpoint's R-group projector.

The index is only valid for the checkpoint that built it.  ``__meta__.json``
records that checkpoint's path and mtime so a mismatch is detectable rather than
silently wrong.

Example
-------
::

    python build_library.py \\
      --checkpoint /home/mogan/checkpoints/flavor_macfrag_v1_best.pt \\
      --corpus /home/mogan/preprocessed/molplatte/flavordb_full/macfrag \\
      --output /home/mogan/libraries/flavor_macfrag_v1

Querying it afterwards::

    import faiss, json, numpy as np
    index = faiss.read_index("<output>/embeddings.faiss")
    rows  = json.load(open("<output>/rows.json"))
    scores, ids = index.search(query_vectors, 100)
    [rows["smiles"][i] for i in ids[0]]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_modules.rgroup_vocab import RGroupLibraryVocab  # noqa: E402
from nnet_modules import MolPLAtte  # noqa: E402


def _load_model(checkpoint: Path, config_path: Optional[Path], device: str):
    """Rebuild the model and load the saved nnet weights.

    ``SaveBestModelCheckpoint`` writes the bare nnet ``state_dict`` (not the
    Lightning or loss wrappers), so the architecture has to be reconstructed from
    the run's Hydra config before loading.
    """
    kwargs: dict = {}
    if config_path is not None:
        from omegaconf import OmegaConf

        config = OmegaConf.load(config_path)
        kwargs = OmegaConf.to_container(config.nnet_module_kwargs, resolve=True)

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Tolerate Lightning-prefixed keys if someone points this at a .ckpt.
    state = {k.replace("model.model.", "").replace("module.", ""): v for k, v in state.items()}

    model = MolPLAtte(**kwargs)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys, e.g. {missing[:3]}", flush=True)
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}", flush=True)
    return model.to(device).eval()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--corpus", required=True, help="corpus holding rgroup_vocab.pkl.gz"
    )
    parser.add_argument("--vocab-path", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="the run's .hydra/config.yaml; without it the model is built with "
        "default hyperparameters, which will mismatch a non-default checkpoint",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    corpus = Path(args.corpus)
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    vocab_path = Path(args.vocab_path) if args.vocab_path else corpus / "rgroup_vocab.pkl.gz"

    if not checkpoint.is_file():
        print(f"ERROR: no checkpoint at {checkpoint}", file=sys.stderr)
        return 1
    if not vocab_path.is_file():
        print(
            f"ERROR: no R-group vocabulary at {vocab_path}.\n"
            f"Build it first:\n"
            f"  python ../../molplatte_preprocess/enumerate_rgroups.py "
            f"--corpus {corpus}",
            file=sys.stderr,
        )
        return 1

    vocab = RGroupLibraryVocab(vocab_path, max_entries=args.max_entries)
    for key, value in [
        ("checkpoint", str(checkpoint)),
        ("vocabulary", str(vocab_path)),
        ("rows", f"{len(vocab):,}"),
        ("effective size", f"{vocab.effective_size():,.0f}"),
        ("device", args.device),
        ("output", str(output)),
    ]:
        print(f"[info] {key:16s} : {value}", flush=True)

    model = _load_model(checkpoint, Path(args.config) if args.config else None, args.device)

    started = time.time()
    chunks = []
    with torch.no_grad():
        for i, graph_batch in enumerate(vocab.batches(args.batch_size)):
            chunks.append(
                model.encode_rgroups(graph_batch.to(args.device)).float().cpu()
            )
            if (i + 1) % 20 == 0:
                done = min((i + 1) * args.batch_size, len(vocab))
                print(f"[prog] embedded {done:,}/{len(vocab):,}", flush=True)

    embeddings = torch.nn.functional.normalize(torch.cat(chunks), dim=-1)
    embeddings = embeddings.numpy().astype(np.float32)
    elapsed = time.time() - started

    import faiss

    output.mkdir(parents=True, exist_ok=True)
    # IndexFlatIP over L2-normalised rows == exact cosine similarity. Exact, not
    # approximate: a library of this size does not need HNSW, and an approximate
    # index would make the retrieval metric depend on index hyperparameters.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(output / "embeddings.faiss"))

    (output / "rows.json").write_text(
        json.dumps(
            {
                "hashes": vocab.hashes,
                "smiles": vocab.smiles,
                "counts": vocab.counts.tolist(),
            }
        )
    )
    (output / "__meta__.json").write_text(
        json.dumps(
            {
                "n_rows": int(embeddings.shape[0]),
                "dim": int(embeddings.shape[1]),
                "index_type": "IndexFlatIP",
                "normalized": True,
                "checkpoint": str(checkpoint),
                "checkpoint_mtime": datetime.fromtimestamp(
                    checkpoint.stat().st_mtime
                ).strftime("%Y-%m-%dT%H:%M:%S"),
                "config": str(args.config) if args.config else None,
                "corpus": str(corpus),
                "vocab_path": str(vocab_path),
                "vocab_provenance": vocab.provenance,
                "effective_size": vocab.effective_size(),
                "embed_seconds": round(elapsed, 2),
                "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
    )

    print(
        f"\n[done] embedded {embeddings.shape[0]:,} R-groups "
        f"(dim {embeddings.shape[1]}) in {elapsed:.1f}s\n"
        f"[done] library : {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
