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
    """Read loss_rgroup_kwargs.temperature independently of the shared module.

    Deliberately a SECOND implementation. Asserting retrieval_scoring against
    itself would pass no matter what it returned.
    """
    text = CONFIG.read_text()
    block = text.split("loss_rgroup_kwargs:", 1)
    assert len(block) == 2, f"loss_rgroup_kwargs not found in {CONFIG}"
    m = re.search(r"temperature:\s*([0-9.eE+-]+)", block[1])
    assert m, "no temperature under loss_rgroup_kwargs"
    return float(m.group(1))


def test_shared_module_reads_the_trained_temperature():
    from retrieval_scoring import rgroup_temperature

    assert rgroup_temperature() == pytest.approx(_rgroup_temperature())


def test_callback_and_inference_score_identically():
    """The two callers must produce the same ranking from the same inputs.

    This is the property that actually matters. Both previously had their own
    copy of `sim/tau + coef*log p`, and both were wrong at different times.
    """
    import numpy as np
    import torch

    from retrieval_scoring import logq_corrected, rgroup_temperature

    rng = np.random.default_rng(0)
    sim = rng.normal(size=(4, 32)).astype(np.float64)
    logp = np.log(rng.random((4, 32)) + 1e-6)
    tau = rgroup_temperature()

    a = logq_corrected(sim, logp, tau, 1.0)                      # callback path
    b = logq_corrected(torch.tensor(sim), torch.tensor(logp),     # inference path
                       tau, 1.0).numpy()
    assert np.allclose(a, b)
    assert np.array_equal(np.argsort(-a, axis=1), np.argsort(-b, axis=1))


def test_formula_is_sim_over_tau_plus_coef_times_log_prior():
    """Pin the arithmetic itself, independently computed.

    The agreement test above compares the two callers against each other, so it
    cannot see a defect in the function they SHARE -- dropping the division, or
    dropping the prior term, moves both sides equally and it still passes. This
    computes the expected value from the formula directly.
    """
    import numpy as np

    from retrieval_scoring import logq_corrected

    rng = np.random.default_rng(1)
    sim = rng.normal(size=(3, 8))
    logp = np.log(rng.random((3, 8)) + 1e-6)
    tau, coef = 0.01, 1.0

    expected = sim / tau + coef * logp
    assert np.allclose(logq_corrected(sim, logp, tau, coef), expected)

    # the temperature must actually divide: a 10x change must move the output
    assert not np.allclose(logq_corrected(sim, logp, tau * 10, coef), expected)
    # the prior must actually be added: coef=0 must differ when logp is nonzero
    assert not np.allclose(logq_corrected(sim, logp, tau, 0.0), expected)
    # and coef=0 must equal pure scaled similarity
    assert np.allclose(logq_corrected(sim, logp, tau, 0.0), sim / tau)


def test_temperature_must_be_positive():
    from retrieval_scoring import logq_corrected

    with pytest.raises(ValueError):
        logq_corrected(1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        logq_corrected(1.0, 0.0, -0.01)


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
    assert sig.parameters["temperature"].default is None, (
        "temperature should default to None so it resolves from the training "
        "config; a literal default is what drifted twice"
    )
    from retrieval_scoring import rgroup_temperature
    default = rgroup_temperature()
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
    assert sig.parameters["temperature"].default is None, (
        "temperature should default to None so it resolves from the config"
    )
    from retrieval_scoring import rgroup_temperature
    default = rgroup_temperature()
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


# --------------------------------------------------------------------------
# Checkpoint self-description
# --------------------------------------------------------------------------

def test_infer_reads_the_sidecar_first():
    import json
    import tempfile

    from checkpoint_spec import infer_model_kwargs, write_sidecar

    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "m.pt"
        ckpt.write_bytes(b"")                       # never read when a sidecar exists
        write_sidecar(ckpt, {"condvec_dim": 1304, "pocket_input_dim": 1280})
        got = infer_model_kwargs({}, ckpt)
        assert got == {"condvec_dim": 1304, "pocket_input_dim": 1280}


def test_infer_detects_an_assembly_head_from_shapes():
    """The exact defect that broke the notebook: kwargs omitted assembly_head."""
    import torch

    from checkpoint_spec import infer_model_kwargs

    sd = {
        "nnet.graph_projector.projection.0.weight": torch.zeros(300, 300),
        # 300 hidden + 24 flavour + 32 reduced pocket = 356
        "nnet.query_projector.projection.0.weight": torch.zeros(300, 356),
        "nnet.assembly_head.fuse.0.weight": torch.zeros(4, 4),
        "nnet.pocket_conditioning.basis": torch.zeros(32, 1280),
    }
    kw = infer_model_kwargs(sd)
    assert kw["assembly_head"] == "AssemblyHead"
    assert kw["pocket_input_dim"] == 1280 and kw["pocket_dim"] == 32
    # condvec_dim is what the CORPUS stores: flavour + RAW pocket, not reduced
    assert kw["condvec_dim"] == 24 + 1280


def test_infer_refuses_to_guess_an_ambiguous_checkpoint():
    """A widened-but-untrained pocket path cannot be told from a wide flavour
    half by shape alone. Guessing loads a model that runs and is wrong."""
    import torch

    from checkpoint_spec import infer_model_kwargs

    sd = {
        "nnet.graph_projector.projection.0.weight": torch.zeros(300, 300),
        "nnet.query_projector.projection.0.weight": torch.zeros(300, 356),
    }
    with pytest.raises(ValueError, match="ambiguous"):
        infer_model_kwargs(sd)
