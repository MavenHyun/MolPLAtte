"""Retrosynthesis planning for generated compounds, via AIZynthFinder.

INFERENCE ONLY. This is deliberately absent from the training path: a tree
search takes seconds to minutes per molecule against milliseconds for SAScore,
and nothing in training needs it.

It runs in its OWN interpreter, and not for tidiness. AIZynthFinder pins
``numpy<2`` and ``rdkit>=2023.9.1,<2024``; this project requires
``rdkit>=2024.3.1``. Those ranges are MUTUALLY EXCLUSIVE -- the two cannot share
an environment at all. And the clash is worse than a failed solve would be:
five feature tables in ``mol_features.py`` store the INDEX into an RDKit enum,
so installing rdkit 2023.x beside the corpus would silently shift the feature
indices of every stored record.

WHAT `solved` MEANS, because it is the number most easily over-read: the search
reached purchasable stock compounds within its time and depth budget. It is not
a claim that a chemist would run the route, and `solved=False` conflates "no
route exists" with "none found in the budget given". Treat it as a filter, not
a verdict.

Against SAScore, which the report already carries: SAScore is a
fragment-contribution heuristic that never attempts a synthesis. This attempts
one. They disagree most on exactly the chemistry this pipeline keeps proposing
-- glycosylations and other decorations that are common in nature and awkward in
a flask.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = ["retro_available", "plan_routes", "RETRO_PYTHON", "RETRO_CONFIG"]

#: Interpreter that can `import aizynthfinder`. Overridable for a different
#: install location.
RETRO_PYTHON = os.environ.get(
    "MOLPLATTE_RETRO_PYTHON",
    str(Path.home() / "miniconda3" / "envs" / "molplatte-retro" / "bin" / "python"))

#: Config written by `download_public_data`, naming the expansion policy,
#: templates and stock.
RETRO_CONFIG = os.environ.get(
    "MOLPLATTE_RETRO_CONFIG",
    str(Path.home() / "datasets" / "aizynthfinder" / "config.yml"))

_SCRIPT = Path(__file__).resolve().parent / "scripts" / "run_retrosynthesis.py"

#: Columns added to the report table.
RETRO_KEYS = ("retro_solved", "retro_steps", "retro_score")


def retro_available() -> bool:
    """True when the interpreter, the script and the config are all present."""
    return (Path(RETRO_PYTHON).is_file() and _SCRIPT.is_file()
            and Path(RETRO_CONFIG).is_file())


def why_unavailable() -> str:
    """A specific reason, so a silent skip can be diagnosed."""
    missing = []
    if not Path(RETRO_PYTHON).is_file():
        missing.append(f"interpreter {RETRO_PYTHON}")
    if not Path(RETRO_CONFIG).is_file():
        missing.append(f"config {RETRO_CONFIG} (run `download_public_data <dir>`)")
    if not _SCRIPT.is_file():
        missing.append(f"runner {_SCRIPT}")
    return "; ".join(missing) or "available"


def plan_routes(smiles: Sequence[str], *, timeout: int = 3600,
                config: Optional[str] = None) -> Dict[str, dict]:
    """Search a route for each unique SMILES.

    Returns ``{smiles: {solved, n_steps, score, n_routes, seconds, error}}``,
    empty when the toolchain is unavailable. Never raises for a molecule: a
    failed search is a recorded error, since one awkward structure must not cost
    the caller the whole table.
    """
    targets = [s for s in dict.fromkeys(smiles) if s]
    if not targets:
        return {}
    if not retro_available():
        logger.warning("[retrosynthesis] unavailable: %s", why_unavailable())
        return {}

    cfg = config or RETRO_CONFIG
    with tempfile.TemporaryDirectory(prefix="molplatte_retro_") as tmp:
        out = Path(tmp) / "routes.json"
        cmd = [RETRO_PYTHON, str(_SCRIPT), str(cfg), str(out), *targets]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("[retrosynthesis] timed out after %ds on %d molecules; "
                           "raise `timeout` or lower the search budget in %s",
                           timeout, len(targets), cfg)
            return {}
        if not out.is_file():
            tail = (proc.stderr or b"").decode(errors="ignore")[-400:]
            logger.warning("[retrosynthesis] produced no output; last stderr:\n%s",
                           tail)
            return {}
        try:
            return json.loads(out.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("[retrosynthesis] unreadable output: %s", exc)
            return {}
