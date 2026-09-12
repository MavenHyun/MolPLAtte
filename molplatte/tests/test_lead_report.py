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
    # Mirrors the real SlotResult. Omitting it is how a table column that reads
    # it broke seven tests at once -- which is the fixture earning its keep.
    core_atoms: tuple = ()
    rgroup_atoms: tuple = ()


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


def test_replaced_atoms_disambiguates_identical_rgroups():
    """Two chemically identical groups at different sites.

    `replaced` is the same SMILES for both -- that is the whole reason the atom
    indices are a column. A tetramethoxyflavone shows "CO" four times, and only
    the site tells them apart.
    """
    from lead_report import build_tables

    results = [
        FakeSlot(0, 0, "core", "CO", [FakeSuggestion(1, 9.0, "*OCC",
                                                     product="CCOc1ccccc1")],
                 rgroup_atoms=(12, 13)),
        FakeSlot(0, 1, "core", "CO", [FakeSuggestion(1, 8.0, "*OCC",
                                                     product="CCOc1ccccc1O")],
                 rgroup_atoms=(21, 22)),
    ]
    table, _ = build_tables(results, "PARENT")
    assert len(table) == 2
    assert set(table["replaced"]) == {"CO"}, "the SMILES cannot distinguish them"
    assert set(table["replaced_atoms"]) == {"12,13", "21,22"}, \
        "the atom indices must"


def test_replaced_atoms_carries_every_site_when_deduped():
    """One product reachable from two sites must name BOTH.

    A symmetric molecule can give the same canonical product from different
    positions. Keeping only the best-scoring site would report one of several
    true answers as if it were the only one.
    """
    from lead_report import build_tables

    same = "CCOc1ccccc1"
    results = [
        FakeSlot(0, 0, "core", "CO", [FakeSuggestion(1, 5.0, "*OCC", product=same)],
                 rgroup_atoms=(21, 22)),
        FakeSlot(0, 1, "core", "CO", [FakeSuggestion(1, 9.0, "*OCC", product=same)],
                 rgroup_atoms=(12, 13)),
    ]
    table, _ = build_tables(results, "PARENT")
    assert len(table) == 1, "one product, one row"
    row = table.iloc[0]
    assert row["n_slots"] == 2 and row["n_sites"] == 2
    # sorted by position, not by which one won on score
    assert row["replaced_atoms"] == "12,13 | 21,22"
    assert row["retrieval_score"] == pytest.approx(9.0), "best score still wins"


def test_replaced_atoms_is_sorted_and_stable():
    from lead_report import build_tables

    results = [FakeSlot(0, 0, "core", "CO",
                        [FakeSuggestion(1, 1.0, "*O", product="CCO")],
                        rgroup_atoms=(22, 21, 12))]
    table, _ = build_tables(results, "PARENT")
    assert table.iloc[0]["replaced_atoms"] == "12,21,22"


# --------------------------------------------------------------------------
# Retrosynthesis columns
# --------------------------------------------------------------------------

def test_retro_columns_absent_unless_requested():
    from lead_report import build_tables

    results = [FakeSlot(0, 0, "core", "CO",
                        [FakeSuggestion(1, 1.0, "*O", product="CCO")])]
    table, _ = build_tables(results, "PARENT")
    for col in ("retro_solved", "retro_steps", "retro_routes"):
        assert col not in table.columns


def test_retro_columns_carry_the_search_result():
    """Including the case that matters: an UNSOLVED molecule.

    SAScore only ever says "harder"; a route search can say "none found", and
    that distinction is the reason this column exists.
    """
    from lead_report import build_tables

    made = "CCO"
    unmade = "CCN"
    results = [FakeSlot(0, 0, "core", "CO", [
        FakeSuggestion(1, 9.0, "*O", product=made),
        FakeSuggestion(2, 8.0, "*N", product=unmade),
    ])]
    retro = {
        made:   {"solved": True,  "n_steps": 2, "score": 0.99, "n_solved_routes": 9},
        unmade: {"solved": False, "n_steps": 6, "score": None, "n_solved_routes": 0},
    }
    table, _ = build_tables(results, "PARENT", None, None, retro)
    rows = {r["product"]: r for r in table.to_dict("records")}
    assert rows[made]["retro_solved"] is True
    assert rows[made]["retro_steps"] == 2 and rows[made]["retro_routes"] == 9
    assert rows[unmade]["retro_solved"] is False
    assert rows[unmade]["retro_routes"] == 0, "unsolved must report zero routes"


def test_retro_missing_molecule_does_not_break_the_row():
    """A molecule the search never reached must not drop its row."""
    from lead_report import build_tables

    results = [FakeSlot(0, 0, "core", "CO",
                        [FakeSuggestion(1, 1.0, "*O", product="CCO")])]
    table, _ = build_tables(results, "PARENT", None, None, {})
    assert len(table) == 1
    assert table.iloc[0]["retro_solved"] is None


def test_score_splits_into_model_and_prior():
    """score = model + prior, exactly. The decomposition is what distinguishes
    'the network prefers this' from 'this fragment is just common'."""
    from lead_report import build_tables

    results = [FakeSlot(0, 0, "core", "CO",
                        [FakeSuggestion(1, 41.59, "*OCC", hash="h1",
                                        corpus_count=5896, product="CCO")])]
    table, _ = build_tables(results, "PARENT", None, None, None,
                            {"h1": -5.635566}, 1.0)
    row = table.iloc[0]
    assert row["prior_term"] == pytest.approx(-5.635566)
    assert row["model_term"] == pytest.approx(41.59 + 5.635566)
    assert row["model_term"] + row["prior_term"] == pytest.approx(41.59)
    # rarity is log10(1/p) in the same units the ranking uses
    assert row["rgroup_idf"] == pytest.approx(5.635566 / 2.302585, abs=1e-4)


def test_count_bands_span_the_skew():
    """Median count is 1 and max is 172,150, so a percentile is degenerate."""
    from lead_report import _count_band

    assert _count_band(1) == "singleton"
    assert _count_band(5) == "rare"
    assert _count_band(50) == "uncommon"
    assert _count_band(500) == "frequent"
    assert _count_band(5000) == "common"
    assert _count_band(172150) == "ubiquitous"


def test_split_columns_absent_without_a_prior():
    from lead_report import build_tables

    results = [FakeSlot(0, 0, "core", "CO",
                        [FakeSuggestion(1, 1.0, "*O", product="CCO")])]
    table, _ = build_tables(results, "PARENT")
    for c in ("model_term", "prior_term", "rgroup_idf"):
        assert c not in table.columns
