"""Work out how to rebuild a model from its checkpoint, instead of being told.

The checkpoints are plain ``state_dict`` files, not Lightning checkpoints, so
they carry no config. Callers therefore had to restate the architecture by hand
-- ``condvec_dim``, ``pocket_input_dim``, ``pocket_dim``, ``assembly_head`` --
and a wrong value is a shape error at best. At worst it is not an error at all:
the notebook shipped for weeks declaring ``condvec_dim=24, pocket_input_dim=0``
against a 356-wide checkpoint, and the first thing that happened when it was
finally executed was a load failure.

Two sources, in order:

1. A **sidecar** ``<checkpoint>.json`` carrying ``model_kwargs``. Written next to
   new checkpoints, so they are self-describing. Authoritative.
2. **Tensor shapes.** Enough for most of it:

     assembly_head     present iff any `assembly_head.*` key exists
     pocket_input_dim  from `pocket_conditioning.project.*` / the PCA basis
     pocket_dim        likewise
     condvec_dim       flavor_dim + pocket_input_dim, where flavor_dim is
                       (query projector in) - (hidden) - pocket_dim

   One case is genuinely ambiguous: a checkpoint whose query projector was
   WIDENED for a pocket that was never trained has no pocket_conditioning
   tensors, so the 56 extra input columns could be 56 flavour bits or 24 flavour
   plus a 32-wide pocket. That is reported as ambiguous rather than guessed --
   guessing would load a model that runs and is wrong.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

__all__ = ["infer_model_kwargs", "sidecar_path", "write_sidecar"]


def sidecar_path(checkpoint: str | Path) -> Path:
    """``foo.pt`` -> ``foo.json``."""
    p = Path(checkpoint)
    return p.with_suffix(".json")


def write_sidecar(checkpoint: str | Path, model_kwargs: Dict,
                  **extra) -> Path:
    """Record how to rebuild this checkpoint, beside it."""
    out = sidecar_path(checkpoint)
    payload = {"model_kwargs": dict(model_kwargs)}
    payload.update(extra)
    if out.exists():
        try:
            old = json.loads(out.read_text())
            old.update(payload)
            payload = old
        except Exception:  # noqa: BLE001 - a corrupt sidecar is not fatal
            pass
    out.write_text(json.dumps(payload, indent=2))
    return out


def _hidden_dim(sd: Dict) -> Optional[int]:
    """Node-embedding width, read off a projector that takes it unmodified."""
    for key in ("nnet.graph_projector.projection.0.weight",
                "nnet.node_projector.projection.0.weight"):
        if key in sd:
            return int(sd[key].shape[1])
    return None


def _pocket_dims(sd: Dict) -> Tuple[Optional[int], Optional[int]]:
    """(pocket_input_dim, pocket_dim) from whichever reduction was trained."""
    basis = sd.get("nnet.pocket_conditioning.basis")
    if basis is not None and hasattr(basis, "shape") and len(basis.shape) == 2:
        return int(basis.shape[1]), int(basis.shape[0])

    first = sd.get("nnet.pocket_conditioning.project.1.weight")
    layers = sorted(k for k in sd
                    if k.startswith("nnet.pocket_conditioning.project.")
                    and k.endswith(".weight"))
    if first is not None and len(first.shape) == 2 and layers:
        return int(first.shape[1]), int(sd[layers[-1]].shape[0])
    return None, None


def infer_model_kwargs(state_dict: Dict,
                       checkpoint: Optional[str | Path] = None) -> Dict:
    """Best-effort model kwargs for ``state_dict``.

    Sidecar wins when present. Raises ValueError when the architecture cannot be
    determined, naming what to pass -- silence here produces a model that loads
    and ranks wrongly.
    """
    if checkpoint is not None:
        side = sidecar_path(checkpoint)
        if side.is_file():
            try:
                spec = json.loads(side.read_text()).get("model_kwargs")
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{side} is not readable JSON: {exc}") from exc
            if spec:
                return dict(spec)

    sd = state_dict.get("state_dict", state_dict)
    kw: Dict = {}

    if any("assembly_head" in k for k in sd):
        kw["assembly_head"] = "AssemblyHead"

    hidden = _hidden_dim(sd)
    qkey = "nnet.query_projector.projection.0.weight"
    if hidden is None or qkey not in sd:
        raise ValueError(
            "cannot read the query projector or hidden width from this "
            "checkpoint; pass model kwargs explicitly"
        )
    query_cond = int(sd[qkey].shape[1]) - hidden

    p_in, p_out = _pocket_dims(sd)
    if p_in is not None:
        if "nnet.pocket_conditioning.basis" in sd:
            # A frozen-PCA reduction. The basis rides in the checkpoint, so the
            # module is built empty-buffered and filled by load_state_dict.
            kw["use_basis"] = True
        flavor = query_cond - p_out
        if flavor < 0:
            raise ValueError(
                f"query projector sees {query_cond} condvec columns but the "
                f"pocket reduction emits {p_out}; the checkpoint is inconsistent"
            )
        kw.update(condvec_dim=flavor + p_in, pocket_input_dim=p_in,
                  pocket_dim=p_out)
        return kw

    if query_cond == 0:
        kw["condvec_dim"] = 0
        return kw

    # No pocket tensors. Either there is no pocket half at all, or there is one
    # that was never trained -- and those need different kwargs.
    raise ValueError(
        f"this checkpoint's query projector takes {query_cond} condition "
        f"columns but carries no pocket_conditioning weights, so it is "
        f"ambiguous: that could be condvec_dim={query_cond} with "
        f"pocket_input_dim=0, or a flavour half plus an UNTRAINED pocket "
        f"reduction. Write a sidecar with write_sidecar(), or pass "
        f"condvec_dim / pocket_input_dim / pocket_dim explicitly."
    )
