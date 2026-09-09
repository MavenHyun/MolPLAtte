"""Guards on constants that inference must share with training.

These are not unit tests of behaviour. They pin the places where a number is
duplicated between the training config and the inference path, because when
those drift nothing raises: the model loads, the SMILES parse, and the ranking
is quietly wrong. Both failures below have actually happened.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIG = SRC / "configs" / "loss_module" / "default.yaml"


def _rgroup_temperature() -> float:
    """Read loss_rgroup_kwargs.temperature without importing hydra."""
    text = CONFIG.read_text()
    block = text.split("loss_rgroup_kwargs:", 1)
    assert len(block) == 2, f"loss_rgroup_kwargs not found in {CONFIG}"
    m = re.search(r"temperature:\s*([0-9.eE+-]+)", block[1])
    assert m, "no temperature under loss_rgroup_kwargs"
    return float(m.group(1))


def test_lead_optimizer_temperature_matches_the_rgroup_loss():
    """Inference must score at the temperature the R-group loss trained at.

    The corrected score is sim/tau + coef * log p(k). tau sets the scale at
    which the similarity and the frequency prior trade off, so a wrong tau does
    not break anything visibly -- it silently reweights them. Shipped history:

      * no division at all (tau = 1.0): the model contributed 0.083 across
        candidates while log p spanned 1.6, so optimize() returned R-groups in
        corpus-count order whatever the conditioning.
      * tau = 0.1: the graph/assembly temperature, read by mistake. Still
        prior-dominated; vanillin's own aldehyde fell out of the top 6.

    The three contrastive terms each carry a DIFFERENT temperature, which is
    what makes this easy to get wrong.
    """
    from lead_optimization import LeadOptimizer

    import inspect

    sig = inspect.signature(LeadOptimizer.__init__)
    default = sig.parameters["temperature"].default
    assert default == pytest.approx(_rgroup_temperature()), (
        f"LeadOptimizer defaults to tau={default} but the R-group loss trains "
        f"at tau={_rgroup_temperature()} ({CONFIG}). Inference would reweight "
        f"similarity against the frequency prior by "
        f"{default / _rgroup_temperature():.0f}x."
    )


def test_retrieval_callback_temperature_matches_the_rgroup_loss():
    """The eval callback applies the same correction and needs the same tau.

    If it drifts, reported Hit@K is computed at a temperature the model was not
    trained at -- comparable across arms that share the bug, but not a number
    that means anything on its own.
    """
    import inspect

    from callbacks.RGroupLibraryRetrieval import RGroupLibraryRetrieval

    sig = inspect.signature(RGroupLibraryRetrieval.__init__)
    default = sig.parameters["temperature"].default
    assert default == pytest.approx(_rgroup_temperature()), (
        f"RGroupLibraryRetrieval defaults to tau={default} but the R-group loss "
        f"trains at tau={_rgroup_temperature()} ({CONFIG})."
    )


def test_the_three_contrastive_temperatures_are_distinct():
    """Pin the thing that makes the above easy to get wrong.

    If these ever become equal the guards above still hold, but the comment
    warning readers off `contrastive.py`'s default would go stale.
    """
    text = CONFIG.read_text()
    found = {
        key: float(re.search(r"temperature:\s*([0-9.eE+-]+)",
                             text.split(f"{key}:", 1)[1]).group(1))
        for key in ("loss_graph_kwargs", "loss_linker_kwargs", "loss_rgroup_kwargs")
        if f"{key}:" in text
    }
    assert len(found) == 3, f"expected three contrastive blocks, found {found}"
    assert len(set(found.values())) > 1, (
        f"temperatures collapsed to one value {found}; the inference-side "
        "comments about reading the wrong one should be revisited"
    )
