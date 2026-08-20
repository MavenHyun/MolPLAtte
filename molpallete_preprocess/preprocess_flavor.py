#!/usr/bin/env python
"""Build a MolPallete pretraining corpus from FlavorDB and/or COCONUT.

One corpus, one or more sources.  ``--source flavordb coconut`` reads both and
writes a single combined corpus, which is what the pretraining runs use: the
flavor compounds are the in-domain set and the natural products supply scale and
chemical diversity.  Building each source separately is still supported and is
how the per-source ablations are produced.

Deduplication
-------------
Duplicates are detected on the **canonical SMILES of the washed molecule** — not
the raw input string, since washing strips salts and neutralises charges, so two
different inputs can be the same molecule.  This catches duplicates *within* a
source as well as across them, and both are common:

* **Across sources**: ~0.8% of COCONUT matches a FlavorDB InChIKey exactly, ~1.9%
  on the connectivity skeleton — the natural-product subset of flavor space.
* **Within FlavorDB**: 25,509 InChIKeys collapse to 13,028 connectivity
  skeletons, dominated by sugars (the sucrose skeleton appears under 283 distinct
  CIDs).  Measured ~4% exact-structure duplicates after washing.

Left in, these inflate the R-group frequency prior and give the retrieval metric
a larger head to memorise.  On a collision the FlavorDB record wins
deterministically — it is in-domain and carries the flavor labels — rather than
whichever worker happened to finish first.  Counts are reported and recorded in
the manifest.  Disable with ``--no-dedup``.

Walks a source of molecules, washes each one, decomposes it into anchored
cores + R-groups, and writes one ``.pt`` per molecule holding the intact graph
plus every decomposition's bookkeeping.  The training repo's dataset samples one
``(decomposition, islinked)`` pair per ``__getitem__`` and materialises the
G / P / R views on the spot -- see ``docs/corpus_format.md`` for why the detached
graphs are not stored.

Examples
--------
Build the flavor corpus (small, fast -- validate the pipeline here first)::

    python preprocess_flavor.py \\
      --source flavordb --method macfrag \\
      --output-path /home/mogan/preprocessed/molpallete/flavordb_full/macfrag \\
      --workers 32 --progress-every 2000

Build the natural-product corpus::

    python preprocess_flavor.py \\
      --source coconut --method macfrag \\
      --output-path /home/mogan/preprocessed/molpallete/coconut_full/macfrag \\
      --max-heavy-atoms 50 --workers 88 --layout hash3 --progress-every 25000

Re-running against a populated directory is refused unless ``--resume`` (skip
molecules already written) or ``--overwrite`` (start clean) is given.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import rdkit
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from molpallete_prep import __version__
from molpallete_prep.anchored_from_partition import partitions_to_decompositions
from molpallete_prep.condvec import CONDVEC_MODES, get_condvec_encoder
from molpallete_prep.decompose import Decomposition, wash
from molpallete_prep.decomposers import (
    family_of,
    get_anchored_decomposer,
    get_fragment_decomposer,
    list_methods,
)
from molpallete_prep.graph_hash import HASH_VERSION, subgraph_hash
from molpallete_prep.graph_ops import detach_rgroups_multi
from molpallete_prep.mol_features import mol_to_pyg
from molpallete_prep.molpla_instance import decomposition_record
from molpallete_prep.preprocess import (
    SHARD_LAYOUTS,
    existing_ids,
    path_for,
    write_manifest,
    write_meta,
    write_record,
)
from molpallete_prep.readers import SizeFilter, read_source

# --------------------------------------------------------------------------- #
# Worker state.  Set once per process by the pool initializer; the alternative
# is shipping the config with every batch, which pickles it thousands of times.
# --------------------------------------------------------------------------- #
_W: Dict[str, object] = {}


def _init_worker(config: dict) -> None:
    """Pool initializer: build the per-process decomposer and condvec encoder."""
    method = config["method"]
    family = family_of(method)
    if family == "anchored":
        decomposer = get_anchored_decomposer(method, **config["method_kwargs"])
    else:
        decomposer = get_fragment_decomposer(method, **config["method_kwargs"])

    _W.clear()
    _W.update(
        config,
        family=family,
        decomposer=decomposer,
        condvec=get_condvec_encoder(config["condvec_mode"]),
    )


def _rgroup_condvec(mol: Chem.Mol, rgroup_atoms: Iterable[int]) -> np.ndarray:
    """Functional-group condition vector for one R-group.

    Extracted as a sub-molecule of the parent rather than from the masked PyG
    graph, because the ``fr_*`` counters need real chemistry and a masked linker
    atom is not a real atom.
    """
    encoder = _W["condvec"]
    atoms = list(rgroup_atoms)
    try:
        smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=atoms, canonical=True)
        submol = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception:
        submol = None
    return encoder.encode(submol)


def _decompose(mol: Chem.Mol) -> List[Decomposition]:
    """Run the configured decomposer and return anchored decompositions."""
    decomposer = _W["decomposer"]
    if _W["family"] == "anchored":
        decompositions = decomposer(mol)
        return list(decompositions)[: _W["max_cores"]]

    partitions = decomposer(mol)
    if not partitions:
        return []
    return partitions_to_decompositions(
        partitions,
        mol.GetNumAtoms(),
        max_cores=_W["max_cores"],
        ratio=_W["core_ratio"],
        max_rgroups=_W["max_rgroups"],
    )


def _process_one(item: Tuple[str, str, str, dict]) -> Tuple[str, str, Optional[dict]]:
    """Wash, decompose and persist one molecule.

    Returns ``(mol_id, status, index_entry)``.  Never raises across the
    multiprocessing boundary -- a single pathological molecule must not abort a
    700K-record build, so every failure becomes a counted status string.
    """
    mol_id, smiles, source, meta = item
    try:
        mol = wash(smiles, remove_stereo=not _W["keep_stereo"],
                   neutralise=_W["neutralise"])
        if mol is None:
            return mol_id, "wash_failed", None

        try:
            mol_data = mol_to_pyg(mol)
        except Exception as exc:
            return mol_id, f"mol_to_pyg_failed:{type(exc).__name__}", None

        try:
            decompositions = _decompose(mol)
        except Exception as exc:
            return mol_id, f"decompose_failed:{type(exc).__name__}", None
        if not decompositions:
            return mol_id, "no_decomp", None

        records = []
        for dec in decompositions:
            record = decomposition_record(dec)
            # The R-group graph depends only on its own atoms plus the masked
            # clone of its core-side neighbour -- NOT on which other R-groups are
            # detached.  So its hash and condition vector are islinked-invariant
            # and can be precomputed here rather than per __getitem__.
            hashes, condvecs = [], []
            for rgroup in dec.rgroups:
                try:
                    _template, detached = detach_rgroups_multi(
                        mol_data,
                        [(rgroup.rgroup_atoms, rgroup.core_linker, rgroup.rgroup_linker)],
                        store_orig=False,
                    )
                    hashes.append(subgraph_hash(detached[0]))
                except Exception:
                    hashes.append("")
                condvecs.append(_rgroup_condvec(mol, rgroup.rgroup_atoms))
            record["rgroup_hashes"] = hashes
            record["rgroup_condvecs"] = np.vstack(condvecs).astype(np.uint8)
            records.append(record)

        payload = {
            "mol_id": mol_id,
            "source": source,
            "smiles": Chem.MolToSmiles(mol),
            "method": _W["method"],
            "n_decomps": len(records),
            "original": mol_data,
            "decompositions": records,
            "meta": meta,
        }
        try:
            write_record(Path(_W["output_path"]), mol_id, payload, _W["layout"])
        except Exception as exc:
            return mol_id, f"write_failed:{type(exc).__name__}", None

        return (
            mol_id,
            "ok",
            {
                "id": mol_id,
                "source": source,
                # Canonical SMILES of the WASHED molecule -- the dedup key. Washing
                # strips salts and neutralises charges, so two different input
                # strings can be the same molecule; the raw input is not a key.
                "canonical": payload["smiles"],
                "n_decomps": len(records),
                "n_rgroups": [r["n_rgroups"] for r in records],
            },
        )
    except Exception as exc:  # noqa: BLE001 -- deliberate catch-all, see docstring
        return mol_id, f"unhandled:{type(exc).__name__}", None


def _process_batch(batch: List[tuple]) -> List[tuple]:
    return [_process_one(item) for item in batch]


def _chunked(iterable: Iterator, size: int) -> Iterator[list]:
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _source_items(args, size_filter: SizeFilter, skip: set) -> Iterator[tuple]:
    """Stream ``(mol_id, smiles, source, meta)`` over every requested source.

    Sources are read in the order given.  FlavorDB is listed first by convention
    so its records enter the pool first, which matters only for readability --
    duplicate resolution is by source name, not by arrival order.
    """
    rng = random.Random(args.sample_seed)
    emitted = 0
    paths = {"flavordb": args.flavordb_path, "coconut": args.coconut_path}
    for source in args.source:
        for record in read_source(
            source, paths.get(source), size_filter=size_filter
        ):
            if args.sample_fraction is not None and rng.random() >= args.sample_fraction:
                continue
            if record.mol_id in skip:
                continue
            yield (record.mol_id, record.smiles, record.source, record.meta)
            emitted += 1
            if args.limit_mols is not None and emitted >= args.limit_mols:
                return


def _build_method_kwargs(args) -> dict:
    """Translate CLI flags into the selected decomposer's keyword arguments."""
    if args.method == "naveja_recap":
        return {
            "ratio": args.core_ratio,
            "include_ring": not args.no_ring_aware,
            "max_cores": args.max_cores,
            "min_rgroup_atoms": args.min_rgroup_atoms,
        }
    if args.method == "bemis_murcko":
        return {}
    return {}  # macfrag / synton take their partition defaults


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_argument_group("source")
    source.add_argument(
        "--source",
        nargs="+",
        choices=("flavordb", "coconut"),
        required=True,
        help="one or more sources; several are merged into a single corpus",
    )
    source.add_argument("--flavordb-path", default=None)
    source.add_argument("--coconut-path", default=None)
    source.add_argument(
        "--no-dedup",
        action="store_true",
        help="keep duplicate molecules (same washed structure). Duplicates inflate "
        "the R-group frequency prior, so this is on by default.",
    )
    source.add_argument("--min-heavy-atoms", type=int, default=5)
    source.add_argument(
        "--max-heavy-atoms",
        type=int,
        default=50,
        help="drop larger molecules as flavor-irrelevant; 50 is ~FlavorDB's p93 "
        "and drops COCONUT's largest 4.4%%. Try 40 for a volatiles-focused corpus.",
    )
    source.add_argument("--limit-mols", type=int, default=None)
    source.add_argument("--sample-fraction", type=float, default=None)
    source.add_argument("--sample-seed", type=int, default=42)

    decomp = parser.add_argument_group("decomposition")
    decomp.add_argument("--method", choices=list_methods(), default="macfrag")
    decomp.add_argument(
        "--core-ratio",
        type=float,
        default=0.5,
        help="a core must hold at least this fraction of the molecule's atoms",
    )
    decomp.add_argument("--max-cores", type=int, default=10)
    decomp.add_argument("--min-rgroup-atoms", type=int, default=1,
                        help="Reject a core if ANY of its R-groups has fewer "
                             "heavy atoms than this. 2 removes single-atom "
                             "R-groups, which otherwise dominate the retrieval "
                             "targets (42 such rows carried 57.3%% of "
                             "occurrences on the ratio-0.4 corpus).")
    decomp.add_argument("--max-rgroups", type=int, default=8)
    decomp.add_argument("--no-ring-aware", action="store_true")
    decomp.add_argument(
        "--keep-stereo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="preserve stereochemistry through washing. On by default because "
        "cis/trans isomerism is chemically load-bearing in flavor -- "
        "(Z)- and (E)-3-hexenol are different odorants.",
    )
    decomp.add_argument(
        "--neutralise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply charge-neutralisation during washing. OFF by default: 4.6%% "
        "of FlavorDB and 2.8%% of COCONUT carry a formal charge (organic acids, "
        "amino acids, quaternary ammonium tastants), and neutralising collapses "
        "formal_charge to one class.",
    )
    decomp.add_argument("--condvec-mode", choices=CONDVEC_MODES, default="neutral")

    out = parser.add_argument_group("output")
    out.add_argument("--output-path", required=True)
    out.add_argument("--layout", choices=list(SHARD_LAYOUTS), default="hash3")
    out.add_argument("--resume", action="store_true", help="skip records already written")
    out.add_argument("--overwrite", action="store_true", help="delete the output first")

    run = parser.add_argument_group("run")
    run.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    run.add_argument("--batch-size", type=int, default=200)
    run.add_argument("--progress-every", type=int, default=5000)

    args = parser.parse_args(argv)
    if args.sample_fraction is not None and not 0.0 < args.sample_fraction <= 1.0:
        parser.error(f"--sample-fraction must be in (0, 1], got {args.sample_fraction}")
    if args.min_heavy_atoms > args.max_heavy_atoms:
        parser.error("--min-heavy-atoms exceeds --max-heavy-atoms")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.output_path)

    if out_path.exists() and any(out_path.iterdir()):
        if args.overwrite:
            print(f"[warn] --overwrite: deleting {out_path}", flush=True)
            shutil.rmtree(out_path)
        elif not args.resume:
            print(
                f"ERROR: {out_path} exists and is not empty. "
                f"Pass --resume to continue it or --overwrite to replace it.",
                file=sys.stderr,
            )
            return 1
    out_path.mkdir(parents=True, exist_ok=True)

    size_filter = SizeFilter(args.min_heavy_atoms, args.max_heavy_atoms)
    method_kwargs = _build_method_kwargs(args)
    condvec_dim = get_condvec_encoder(args.condvec_mode).dim

    skip: set = set()
    prior_index: List[dict] = []
    if args.resume:
        meta_path = out_path / "__meta__.json"
        if meta_path.is_file():
            prior = json.loads(meta_path.read_text())
            prior_index = [
                {"id": i, "n_decomps": d, "n_rgroups": r}
                for i, d, r in zip(
                    prior["ids"], prior["n_decomps"], prior["n_rgroups_per_decomp"]
                )
            ]
            candidates = [entry["id"] for entry in prior_index]
            skip = existing_ids(out_path, candidates, args.layout)
            prior_index = [e for e in prior_index if e["id"] in skip]
        print(f"[resume] {len(skip):,} records already on disk", flush=True)

    worker_config = {
        "method": args.method,
        "method_kwargs": method_kwargs,
        "core_ratio": args.core_ratio,
        "max_cores": args.max_cores,
        "max_rgroups": args.max_rgroups,
        "keep_stereo": args.keep_stereo,
        "neutralise": args.neutralise,
        "condvec_mode": args.condvec_mode,
        "output_path": str(out_path),
        "layout": args.layout,
    }

    for key, value in [
        ("version", __version__),
        ("sources", " + ".join(args.source)),
        ("source paths", {
            k: v for k, v in
            [("flavordb", args.flavordb_path), ("coconut", args.coconut_path)]
            if k in args.source
        } or "<defaults>"),
        ("dedup", not args.no_dedup),
        ("method", f"{args.method} ({family_of(args.method)} family)"),
        ("method_kwargs", method_kwargs),
        ("core_ratio", args.core_ratio),
        ("max_cores", args.max_cores),
        ("max_rgroups", args.max_rgroups),
        ("heavy_atoms", f"[{args.min_heavy_atoms}, {args.max_heavy_atoms}]"),
        ("keep_stereo", args.keep_stereo),
        ("neutralise", args.neutralise),
        ("condvec", f"{args.condvec_mode} (dim {condvec_dim})"),
        ("output_path", str(out_path)),
        ("layout", args.layout),
        ("workers", args.workers),
    ]:
        print(f"[info] {key:16s} : {value}", flush=True)

    index: List[dict] = list(prior_index)
    # canonical SMILES -> position in `index`. Records are written by the worker
    # before the parent sees them, so duplicates are resolved after the fact by
    # deleting the losing file -- cheap, because it is one file per molecule.
    seen_canonical: Dict[str, int] = {}
    n_deduped = 0
    dedup_enabled = not args.no_dedup
    status_counts: Counter = Counter()
    started = time.time()
    last_report, last_time = 0, started

    items = _source_items(args, size_filter, skip)
    batches = _chunked(items, args.batch_size)

    processed = 0
    with mp.Pool(
        args.workers, initializer=_init_worker, initargs=(worker_config,)
    ) as pool:
        for results in pool.imap_unordered(_process_batch, batches, chunksize=1):
            for _mol_id, status, entry in results:
                status_counts[status] += 1
                if entry is not None:
                    if dedup_enabled:
                        key = entry.get("canonical")
                        previous = seen_canonical.get(key) if key else None
                        if previous is not None:
                            # Deterministic winner: FlavorDB is in-domain and
                            # carries the flavor labels, so it beats COCONUT
                            # regardless of which finished first.
                            incumbent = index[previous]
                            if (
                                incumbent.get("source") != "flavordb"
                                and entry.get("source") == "flavordb"
                            ):
                                loser, index[previous] = incumbent, entry
                            else:
                                loser = entry
                            try:
                                path_for(
                                    out_path, loser["id"], args.layout
                                ).unlink(missing_ok=True)
                            except OSError:
                                pass
                            n_deduped += 1
                            status_counts["duplicate_molecule"] += 1
                            processed += 1
                            continue
                        if key:
                            seen_canonical[key] = len(index)
                    index.append(entry)
                processed += 1
            if processed - last_report >= args.progress_every:
                now = time.time()
                rate = processed / max(now - started, 1e-9)
                window = (processed - last_report) / max(now - last_time, 1e-9)
                print(
                    f"[prog] {processed:>9,} mols  elapsed {now - started:8.1f}s  "
                    f"rate {rate:7.1f} mol/s  window {window:7.1f} mol/s  "
                    f"written={len(index):,}  {dict(status_counts.most_common(4))}",
                    flush=True,
                )
                last_report, last_time = processed, now

    elapsed = time.time() - started
    index.sort(key=lambda e: e["id"])

    provenance = {
        "molpallete_prep_version": __version__,
        # Five of the eleven atom/bond feature tables are RDKit enums whose
        # cardinality can shift between releases, which would silently invalidate
        # this corpus's embedding indices. Record the version that built it.
        "rdkit_version": rdkit.__version__,
        "graph_hash_version": HASH_VERSION,
        "sources": list(args.source),
        "source": args.source[0] if len(args.source) == 1 else "+".join(args.source),
        "dedup": dedup_enabled,
        "n_duplicates_removed": n_deduped,
        "method": args.method,
        "method_family": family_of(args.method),
        "decomposition_params": method_kwargs,
        "core_ratio": args.core_ratio,
        "max_cores": args.max_cores,
        "max_rgroups": args.max_rgroups,
        "keep_stereo": args.keep_stereo,
        "neutralise": args.neutralise,
        "size_filter": size_filter.as_dict(),
        "condvec_mode": args.condvec_mode,
        "condvec_dim": condvec_dim,
        "sample_fraction": args.sample_fraction,
        "sample_seed": args.sample_seed,
        "layout": args.layout,
        "params_provenance": "written at preprocessing time by preprocess_flavor.py",
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_meta(
        out_path,
        ids=[e["id"] for e in index],
        n_decomps=[e["n_decomps"] for e in index],
        n_rgroups_per_decomp=[e["n_rgroups"] for e in index],
        provenance=provenance,
    )
    write_manifest(
        out_path,
        {
            **provenance,
            "argv": sys.argv[1:],
            "workers": args.workers,
            "batch_size": args.batch_size,
            "total_mols_processed": processed,
            "n_records": len(index),
            "n_resumed": len(prior_index),
            "n_records_per_source": dict(
                Counter(e.get("source", "unknown") for e in index)
            ),
            "status_counts": dict(status_counts),
            "elapsed_seconds": round(elapsed, 2),
            "throughput_mol_per_s": round(processed / max(elapsed, 1e-9), 2),
            "completed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    total_decomps = sum(e["n_decomps"] for e in index)
    total_rgroups = sum(sum(e["n_rgroups"]) for e in index)
    print(
        f"\n[done] processed {processed:,} in {elapsed:.1f}s "
        f"({processed / max(elapsed, 1e-9):.1f} mol/s)\n"
        f"[done] records written    : {len(index):,}\n"
        + (
            f"[done] per source         : "
            f"{dict(Counter(e.get('source', 'unknown') for e in index))}\n"
            if len(args.source) > 1
            else ""
        )
        + (
            f"[done] duplicates removed : {n_deduped:,} "
            f"(same washed structure; FlavorDB kept on collision)\n"
            if dedup_enabled
            else ""
        ) +
        f"[done] decompositions     : {total_decomps:,} "
        f"({total_decomps / max(len(index), 1):.2f} per molecule)\n"
        f"[done] R-groups           : {total_rgroups:,} "
        f"({total_rgroups / max(total_decomps, 1):.2f} per decomposition)\n"
        f"[done] status             : {dict(status_counts)}\n"
        f"[done] corpus             : {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
