#!/usr/bin/env python
"""Retrosynthesis search for a list of SMILES, as JSON on stdout.

Runs in its OWN interpreter. AIZynthFinder pins numpy<2 and rdkit<2024, while
this project needs rdkit>=2024.3.1 -- those ranges are MUTUALLY EXCLUSIVE, so
the two cannot share an environment. Worse than a version clash: the corpus
stores INDICES into RDKit enums, so installing rdkit 2023.x beside it would
silently shift the feature indices of every stored record.

    python run_retrosynthesis.py <config.yml> <out.json> <smiles> [smiles ...]

Per molecule the result carries:

    solved       a route to purchasable stock was found
    n_steps      reaction steps in the best route (None if unsolved)
    score        AiZynthFinder's route score for the best route
    n_routes     distinct routes found
    top_route    the best route as a nested dict, for inspection

`solved` is the number that matters and the one most easily over-read: it means
the search reached purchasable building blocks within its time and depth limits.
It is not a claim that a chemist would run the route, and "unsolved" can mean
"no route exists" or merely "not found in the budget given".
"""
import json
import sys
import time


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    config, out_path, smiles = sys.argv[1], sys.argv[2], sys.argv[3:]

    from aizynthfinder.aizynthfinder import AiZynthFinder

    finder = AiZynthFinder(configfile=config)
    finder.stock.select(finder.stock.items[0])
    finder.expansion_policy.select(finder.expansion_policy.items[0])
    if finder.filter_policy.items:
        finder.filter_policy.select(finder.filter_policy.items[0])

    results = {}
    for smi in smiles:
        rec = {"solved": None, "n_steps": None, "score": None,
               "n_routes": 0, "seconds": None, "error": None}
        t0 = time.time()
        try:
            finder.target_smiles = smi
            finder.tree_search()
            finder.build_routes()
            stats = finder.extract_statistics()
            rec["solved"] = bool(stats.get("is_solved"))
            rec["n_routes"] = int(stats.get("number_of_routes") or 0)
            # Discriminative where `score` is not: the best-route state score
            # saturates near 0.998 for anything solved, so it separates nothing.
            rec["n_solved_routes"] = int(stats.get("number_of_solved_routes") or 0)
            # number_of_steps is the depth of the best route; absent when the
            # search solved nothing, so it is left as None rather than 0 --
            # zero steps would read as "already purchasable".
            steps = stats.get("number_of_steps")
            rec["n_steps"] = int(steps) if steps not in (None, "") else None
            # v4 returns a list of DICTS ({'state score': 0.99}), not floats,
            # so max() over them raises TypeError comparing dict to dict. The
            # earlier fields survive that because they are set first, which is
            # how this shipped looking like it worked.
            scores = list(finder.routes.scores) if finder.routes else []
            vals = []
            for entry in scores:
                if isinstance(entry, dict):
                    vals += [float(v) for v in entry.values()
                             if isinstance(v, (int, float))]
                elif isinstance(entry, (int, float)):
                    vals.append(float(entry))
            rec["score"] = max(vals) if vals else None
        except Exception as exc:  # noqa: BLE001 - one bad molecule must not
            rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rec["seconds"] = round(time.time() - t0, 1)
        results[smi] = rec
        print(f"[retro] {smi[:60]:<60} solved={rec['solved']} "
              f"steps={rec['n_steps']} {rec['seconds']}s", file=sys.stderr)

    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
