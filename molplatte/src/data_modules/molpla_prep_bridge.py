"""Single import surface for everything borrowed from ``molplatte_prep``.

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
    "load_vocabulary",
    "rgroup_smiles",
    "sample_islinked",
    "subgraph_hash",
]


def _ensure_importable() -> None:
    try:
        import molplatte_prep  # noqa: F401
        return
    except ImportError:
        pass

    env = os.environ.get("MOLPLATTE_PREP_PATH")
    candidates = [Path(env)] if env else []
    # …/MolPLAtte/molplatte/src/data_modules -> …/MolPLAtte/molplatte_preprocess
    # src-layout: the package now lives under molplatte_preprocess/src/
    candidates.append(Path(__file__).resolve().parents[3] / "molplatte_preprocess" / "src")
    candidates.append(Path(__file__).resolve().parents[3] / "molplatte_preprocess")

    for candidate in candidates:
        if (candidate / "molplatte_prep" / "__init__.py").is_file():
            sys.path.insert(0, str(candidate))
            return

    raise ImportError(
        "cannot import 'molplatte_prep'. Install the preprocessing repo "
        "(`pip install -e ../molplatte_preprocess`) or set MOLPLATTE_PREP_PATH "
        f"to the directory containing it. Tried: {[str(c) for c in candidates]}"
    )


_ensure_importable()

from molplatte_prep.decompose import Decomposition, RGroupInfo  # noqa: E402
from molplatte_prep.graph_hash import subgraph_hash  # noqa: E402
from molplatte_prep.lmdb_store import hydrate  # noqa: E402
from molplatte_prep.molpla_instance import build_instance, sample_islinked  # noqa: E402
from molplatte_prep.preprocess import path_for  # noqa: E402
from molplatte_prep.rgroup_library import load_vocabulary, rgroup_smiles  # noqa: E402
