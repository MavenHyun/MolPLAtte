"""The one place R-group retrieval scores are computed.

Training optimises InfoNCE, whose critic converges on PMI --
``log p(k|q) - log p(k)`` -- not on the posterior. Ranking by similarity alone
therefore systematically favours rare R-groups, and adding ``log p(k)`` back is
what makes the ranking comparable to a frequency prior. The corrected score is

    sim / tau  +  coef * log p(k)

and ``tau`` is load-bearing: it sets the scale at which the similarity term and
the frequency prior trade off. A wrong ``tau`` breaks nothing visibly. The model
loads, the SMILES parse, Hit@K is still a number -- the two terms are simply
reweighted, and the ranking quietly becomes something else.

This module exists because that formula used to live in two places, and the
constant in it lived in three. Both copies were wrong at different times on
2026-09-09:

* ``LeadOptimizer`` omitted the division entirely (tau = 1.0). Measured on the
  shipped checkpoint, the model contributed 0.083 across candidates while
  ``log p`` spanned 1.6, so ``optimize()`` returned R-groups in corpus-count
  order no matter what conditioning was supplied -- while the retrieval callback
  reported 13x lift over that same prior.
* The first repair then used tau = 0.1, which is the graph-contrastive and
  assembly temperature. The three contrastive terms each carry their OWN
  (graph 0.1, linker 0.05, rgroup 0.01), so reading any but ``loss_rgroup_kwargs``
  is off by 5x or 10x.

So the temperature is read from the training config rather than restated, and
both callers share ``logq_corrected``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

__all__ = ["logq_corrected", "rgroup_temperature", "RGROUP_TEMPERATURE_CONFIG"]

#: The config that sets the R-group contrastive loss temperature. This file is
#: the single source of truth; nothing downstream should restate the number.
RGROUP_TEMPERATURE_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "loss_module" / "default.yaml"
)

_CACHE: Optional[float] = None


def rgroup_temperature(config_path: Optional[Path] = None) -> float:
    """Return ``loss_rgroup_kwargs.temperature`` from the training config.

    Parsed with a regex rather than through hydra/omegaconf so that inference
    and tests can call it without composing a config or importing hydra.

    Raises rather than falling back to a literal: a silent default is exactly
    how this value drifted from training in the first place.
    """
    global _CACHE
    if config_path is None and _CACHE is not None:
        return _CACHE

    path = Path(config_path) if config_path else RGROUP_TEMPERATURE_CONFIG
    if not path.is_file():
        raise FileNotFoundError(
            f"cannot read the R-group loss temperature: {path} does not exist. "
            "Pass config_path, or pass an explicit temperature to the caller."
        )
    text = path.read_text()
    if "loss_rgroup_kwargs:" not in text:
        raise KeyError(f"no loss_rgroup_kwargs block in {path}")
    tail = text.split("loss_rgroup_kwargs:", 1)[1]
    match = re.search(r"temperature:\s*([0-9.eE+-]+)", tail)
    if not match:
        raise KeyError(f"no temperature under loss_rgroup_kwargs in {path}")
    value = float(match.group(1))
    if not value > 0:
        raise ValueError(f"temperature must be positive, got {value} from {path}")
    if config_path is None:
        _CACHE = value
    return value


def logq_corrected(similarity, log_prior, temperature: float,
                   popularity_coef: float = 1.0):
    """``similarity / temperature + popularity_coef * log_prior``.

    Elementwise and backend-agnostic: both operands may be numpy arrays or torch
    tensors, as long as they broadcast against each other. The callback passes
    the similarity top-k and the matching slice of the prior; inference passes
    the full library. Neither needs to know what the other does.

    ``popularity_coef`` of 0 disables the correction and leaves a pure
    similarity ranking, which is kept because ``hit@K`` is logged both ways.
    """
    if not temperature > 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = similarity / temperature
    if not popularity_coef:
        return scaled
    return scaled + popularity_coef * log_prior
