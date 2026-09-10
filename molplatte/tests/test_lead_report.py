"""Guards on the report layer.

The dedup is the part that can silently lie: `optimize` returns one result per
(decomposition, slot), so without it the same molecule proposed from three
slots counts as three compounds and the table overstates what the model made.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass
class FakeSuggestion:
    rank: int
    score: float
    smiles: str
    hash: str = "h"
    corpus_count: int = 0
    is_novel: bool = False
    product: Optional[str] = None
    product_error: str = ""
    aromaticity_kept: Optional[bool] = True


@dataclass
class FakeSlot:
    decomp_index: int
    slot_index: int
    core_smiles: str
    original_rgroup: str
    suggestions: List[FakeSuggestion] = field(default_factory=list)


def test_same_product_from_two_slots_is_one_compound():
    from lead_report import build_tables

    dup = "CCO"
    results = [
        FakeSlot(0, 0, "C", "*C", [FakeSuggestion(1, 10.0, "*O", product=dup)]),
        FakeSlot(1, 0, "C", "*C", [FakeSuggestion(1, 42.0, "*O", product=dup)]),
    ]
    table, failures = build_tables(results, "CCC")
    assert len(table) == 1, "the same product from two slots must collapse"
    row = table.iloc[0]
    assert row["n_slots"] == 2
    assert row["retrieval_score"] == pytest.approx(42.0), "best score must win"
    assert set(row["found_in_slots"].split(",")) == {"d0/s0", "d1/s0"}
    assert len(failures) == 0


def test_unassembled_suggestions_go_to_failures_with_a_reason():
    from lead_report import build_tables

    results = [FakeSlot(0, 0, "C", "*C", [
        FakeSuggestion(1, 5.0, "*O", product=None, product_error="unsanitisable"),
        FakeSuggestion(2, 4.0, "*N", product="CCN"),
    ])]
    table, failures = build_tables(results, "CCC")
    assert len(table) == 1 and len(failures) == 1
    assert failures.iloc[0]["reason"] == "unsanitisable"


def test_table_is_sorted_by_retrieval_score():
    from lead_report import build_tables

    results = [FakeSlot(0, 0, "C", "*C", [
        FakeSuggestion(1, 1.0, "*O", product="CCO"),
        FakeSuggestion(2, 9.0, "*N", product="CCN"),
    ])]
    table, _ = build_tables(results, "CCC")
    assert list(table["product"]) == ["CCN", "CCO"]


def test_input_compound_is_flagged_when_recovered():
    from lead_report import build_tables

    results = [FakeSlot(0, 0, "C", "*C",
                        [FakeSuggestion(1, 1.0, "*O", product="CCC")])]
    table, _ = build_tables(results, "CCC")
    assert bool(table.iloc[0]["is_input"]) is True


def test_flavor_validation_names_the_bad_label():
    from lead_report import PERMISSIBLE_FLAVORS, check_flavors

    assert check_flavors(["sweet", "woody"]) == ["sweet", "woody"]
    assert len(PERMISSIBLE_FLAVORS) == 24
    with pytest.raises(ValueError, match="banana"):
        check_flavors(["sweet", "banana"])


def test_accepts_both_smiles_and_mol():
    from rdkit import Chem

    from lead_report import as_mol

    _, a = as_mol("OCC")
    _, b = as_mol(Chem.MolFromSmiles("OCC"))
    assert a == b == "CCO", "both inputs must canonicalise identically"
    with pytest.raises(ValueError):
        as_mol("not a molecule %%%")
    with pytest.raises(TypeError):
        as_mol(42)


def test_reference_specs_and_deltas():
    """The input's own specs, and each product's change against them."""
    from lead_report import build_tables, molecule_scores

    ethanol = "CCO"
    ref = molecule_scores(ethanol)
    assert ref["MW"] == pytest.approx(46.069, abs=1e-2)

    results = [FakeSlot(0, 0, "C", "*C",
                        [FakeSuggestion(1, 5.0, "*O", product="CCCO")])]
    table, _ = build_tables(results, ethanol, ref)
    row = table.iloc[0]
    assert row["dMW"] == pytest.approx(row["MW"] - ref["MW"])
    assert row["dMW"] > 0, "propanol is heavier than ethanol"


def test_recovering_the_input_gives_exactly_zero_deltas():
    """A free correctness check: if the model returns the input compound, every
    delta must be 0. A non-zero one means the reference and the product were
    scored differently."""
    from lead_report import build_tables, molecule_scores

    smi = "COc1cc(C=O)ccc1O"
    ref = molecule_scores(smi)
    results = [FakeSlot(0, 0, "C", "*C",
                        [FakeSuggestion(1, 5.0, "*O", product=smi)])]
    table, _ = build_tables(results, smi, ref)
    row = table.iloc[0]
    for k in ("MW", "logP", "QED", "SAScore", "NPScore"):
        assert row[f"d{k}"] == pytest.approx(0.0, abs=1e-9), f"d{k} must be 0"
    assert bool(row["is_input"]) is True


def test_delta_columns_absent_without_a_reference():
    from lead_report import build_tables

    results = [FakeSlot(0, 0, "C", "*C",
                        [FakeSuggestion(1, 5.0, "*O", product="CCO")])]
    table, _ = build_tables(results, "CCC")
    assert "dMW" not in table.columns
