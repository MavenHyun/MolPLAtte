"""Single import surface for everything borrowed from ``molpallete_prep``.

The preprocessing package is a sibling repo rather than a published dependency,
so import failures are common during setup.  Funnelling every cross-repo import
through one module means the fix-up (and the error message that explains it)
lives in exactly one place.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "Decomposition",
    "RGroupInfo",
    "build_instance",
    "hydrate",
    "path_for",
    "sample_islinked",
    "subgraph_hash",
]


def _ensure_importable() -> None:
    try:
        import molpallete_prep  # noqa: F401
        return
    except ImportError:
        pass

    env = os.environ.get("MOLPALLETE_PREP_PATH")
    candidates = [Path(env)] if env else []
    # …/MolPallete/molpallete/src/data_modules -> …/MolPallete/molpallete_preprocess
    candidates.append(Path(__file__).resolve().parents[3] / "molpallete_preprocess")

    for candidate in candidates:
        if (candidate / "molpallete_prep" / "__init__.py").is_file():
            sys.path.insert(0, str(candidate))
            return

    raise ImportError(
        "cannot import 'molpallete_prep'. Install the preprocessing repo "
        "(`pip install -e ../molpallete_preprocess`) or set MOLPALLETE_PREP_PATH "
        f"to the directory containing it. Tried: {[str(c) for c in candidates]}"
    )


_ensure_importable()

from molpallete_prep.decompose import Decomposition, RGroupInfo  # noqa: E402
from molpallete_prep.graph_hash import subgraph_hash  # noqa: E402
from molpallete_prep.lmdb_store import hydrate  # noqa: E402
from molpallete_prep.molpla_instance import build_instance, sample_islinked  # noqa: E402
from molpallete_prep.preprocess import path_for  # noqa: E402
