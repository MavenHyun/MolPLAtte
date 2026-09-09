"""Regression guards for failures that produced no error.

Every case here corresponds to a bug that ran to completion and wrote
plausible-looking output. None of them would have been caught by "does it
crash?", which is why they survived long enough to reach a corpus:

- the flavor condvec collapsed to a single MW>350 bit across all four corpora
  and nothing failed; the vector was simply uninformative (24b75d1)
- CrossDocked's official test set would have leaked into train through
  ligand-level deduplication, and training would have looked fine (83e7601)
- a drug-like heavy-atom floor silently rejected the cognate ligand of the only
  human olfactory receptor structure in the PDB (233c52c)

Run: pytest molplatte_preprocess/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
#: The training package, derived from this file rather than from $HOME. A
#: hardcoded ~/github/MolPLAtte path passes on the machine that wrote it and
#: fails on any checkout elsewhere -- which is exactly what CI is for, and is
#: how this was found.
TRAIN_SRC = Path(__file__).resolve().parents[2] / "molplatte" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rdkit import Chem  # noqa: E402

from molplatte_prep.condvec import (  # noqa: E402
    CONDVEC_VERSION,
    FLAVOR_LABELS,
    FlavorCondVec,
)
from molplatte_prep.pocket_ligands import (  # noqa: E402
    CHEMOSENSORY_COGNATE,
    MIN_HEAVY_ATOMS,
    assess_ligand,
    ccd_code_from_path,
    is_artifact,
)
from molplatte_prep.readers import _xd_split  # noqa: E402


def bits(vec):
    return {FLAVOR_LABELS[i] for i, x in enumerate(vec) if x > 0.5}


# ---------------------------------------------------------------- condvec
class TestFlavorCondvecReadsItsTables:
    """The condvec must consult the label tables, not fall through to physics.

    The original bug read mol_context off a record whose `meta` key did not
    exist yet, so the InChIKey was always "" and every molecule took the
    MW>350 branch. Asserting "a vector was produced" would have passed.
    """

    @pytest.fixture
    def encoder(self):
        return FlavorCondVec(
            measured={"KEY-MEASURED": ["bitter"], "FDB1": ["sweet"]},
            mined={"KEY-MINED": ["fruity"]},
        )

    def test_resolves_measured_label_by_inchikey(self, encoder):
        v = encoder.encode(None, {"inchikey": "KEY-MEASURED", "mol_id": "", "mw": None})
        assert bits(v) == {"bitter"}

    def test_resolves_measured_label_by_mol_id(self, encoder):
        v = encoder.encode(None, {"inchikey": "", "mol_id": "FDB1", "mw": None})
        assert bits(v) == {"sweet"}

    def test_measurement_outranks_the_mw_physics_rule(self, encoder):
        """A heavy compound with a measured label is NOT odorless.

        This is the exact shape of the observed failure: FDB18250360, a sweet
        glycoside at MW 508, was stored as `odorless`.
        """
        v = encoder.encode(None, {"inchikey": "KEY-MEASURED", "mol_id": "", "mw": 508.0})
        assert bits(v) == {"bitter"}
        assert "odorless" not in bits(v)

    def test_mined_labels_are_used_when_unmeasured(self, encoder):
        v = encoder.encode(None, {"inchikey": "KEY-MINED", "mol_id": "", "mw": None})
        assert bits(v) == {"fruity"}

    def test_unknown_light_molecule_is_unknown_not_odorless(self, encoder):
        v = encoder.encode(None, {"inchikey": "NOPE", "mol_id": "", "mw": 120.0})
        assert bits(v) == {"unknown"}

    def test_unknown_heavy_molecule_is_odorless(self, encoder):
        """`odorless` is a positive volatility claim, distinct from `unknown`."""
        v = encoder.encode(None, {"inchikey": "NOPE", "mol_id": "", "mw": 800.0})
        assert bits(v) == {"odorless"}

    def test_vector_is_never_all_zero(self, encoder):
        for ctx in ({"inchikey": "KEY-MEASURED"}, {"inchikey": "X", "mw": 90.0},
                    {"inchikey": "X", "mw": 900.0}, {}):
            assert bits(encoder.encode(None, ctx)), f"all-zero vector for {ctx}"

    def test_condvec_version_is_past_the_broken_build(self):
        """v1 corpora carry the MW-only vector and must stay distinguishable."""
        assert CONDVEC_VERSION >= 2


class TestOdorlessIsNotAssertedAlongsideAnOdour:
    """`odorless` claims nothing is smelled; an odour class contradicts it.

    145 rows of flavor_measured.jsonl (1.3%) assert both -- a merge artifact,
    e.g. FDB72 ['medicinal', 'odorless', 'woody']. Nothing failed: the vector
    simply carried two incompatible bits and the model had to learn around it.

    Taste is a separate axis. A taste receptor sits in solution, so sucrose is
    correctly both `odorless` and `sweet`, and clearing `odorless` there would
    destroy real information.
    """

    @pytest.fixture
    def encoder(self):
        return FlavorCondVec(measured={
            "ODOUR": ["medicinal", "odorless", "woody"],
            "TASTE": ["odorless", "sweet"],
            "BOTH": ["odorless", "sweet", "woody"],
            "PLAIN": ["odorless"],
        })

    def test_odour_class_clears_odorless(self, encoder):
        got = bits(encoder.encode(None, {"inchikey": "ODOUR"}))
        assert got == {"medicinal", "woody"}

    def test_taste_does_not_clear_odorless(self, encoder):
        """Sucrose is odorless and sweet. Both bits are correct."""
        got = bits(encoder.encode(None, {"inchikey": "TASTE"}))
        assert got == {"odorless", "sweet"}

    def test_odour_wins_even_when_a_taste_is_also_present(self, encoder):
        got = bits(encoder.encode(None, {"inchikey": "BOTH"}))
        assert got == {"sweet", "woody"}

    def test_odorless_alone_survives(self, encoder):
        assert bits(encoder.encode(None, {"inchikey": "PLAIN"})) == {"odorless"}

    def test_taste_and_odour_label_sets_partition_the_vocabulary(self):
        from molplatte_prep.condvec import ODOUR_LABELS, TASTE_LABELS

        assert not (TASTE_LABELS & ODOUR_LABELS)
        assert TASTE_LABELS | ODOUR_LABELS | {"odorless", "unknown"} == set(FLAVOR_LABELS)


class TestMolContextKeySpellings:
    """COCONUT uses snake_case, FlavorDB uses PubChem CamelCase.

    Reading only one spelling yields "" for every molecule from the other
    source -- which is half the corpus, silently unlabelled.
    """

    @pytest.fixture
    def mol_context(self):
        from preprocess_flavor import _mol_context

        return _mol_context

    def test_reads_coconut_spelling(self, mol_context):
        ctx = mol_context("CNP1", {"standard_inchi_key": "AAA", "molecular_weight": "200"})
        assert ctx["inchikey"] == "AAA"
        assert ctx["mw"] == "200"

    def test_reads_flavordb_spelling(self, mol_context):
        ctx = mol_context("FDB1", {"InChIKey": "BBB", "MolecularWeight": "300"})
        assert ctx["inchikey"] == "BBB"
        assert ctx["mw"] == "300"

    def test_missing_meta_does_not_raise(self, mol_context):
        assert mol_context("X", None)["inchikey"] == ""


# ---------------------------------------------------------------- split leak
class TestCrossDockedSplitPrecedence:
    """86 ligands bind both train and test pockets.

    One record per ligand means the ligand needs ONE split. Taking the first
    key's split -- the obvious implementation -- puts those in train and leaks
    the official test set. Training would have looked entirely normal.
    """

    SPLIT = {1: "train", 2: "test", 3: "train", 4: "val"}

    def test_test_wins_over_train(self):
        assert _xd_split(["1", "2", "3"], self.SPLIT) == "test"

    def test_test_wins_regardless_of_key_order(self):
        assert _xd_split(["3", "1", "2"], self.SPLIT) == "test"

    def test_val_wins_over_train(self):
        assert _xd_split(["1", "4"], self.SPLIT) == "val"

    def test_train_when_only_train(self):
        assert _xd_split(["1", "3"], self.SPLIT) == "train"

    def test_unassigned_when_absent_from_split(self):
        assert _xd_split(["99"], self.SPLIT) == "unassigned"

    def test_non_numeric_keys_do_not_raise(self):
        assert _xd_split(["abc"], self.SPLIT) == "unassigned"


class TestCcdCodeParsing:
    def test_parses_crossdocked_filename(self):
        p = "3HAO_CUPMC_1_172_0/4hvr_A_rec_1yfy_3ha_lig_tt_min_0.sdf"
        assert ccd_code_from_path(p) == "3HA"

    def test_returns_none_when_absent(self):
        assert ccd_code_from_path("nonsense.sdf") is None


# ---------------------------------------------------------------- ligands
class TestLigandValidation:
    ESTRONE = "C[C@]12CCc3c(ccc4cc(O)ccc34)[C@@H]1CCC2=O"
    PROPIONATE = "CCC(=O)O"

    def test_blocklist_categories(self):
        assert is_artifact("ADP") == "cofactor"
        assert is_artifact("MRD") == "cryo"
        assert is_artifact("PPV") == "buffer"
        assert is_artifact("HOH") == "water"
        assert is_artifact("ZN") == "ion"
        assert is_artifact("NAG") == "glycan"
        assert is_artifact("3HA") is None

    def test_cofactors_can_be_readmitted(self):
        assert is_artifact("ADP", keep_cofactors=True) is None
        # re-admitting cofactors must not loosen any other category
        assert is_artifact("MRD", keep_cofactors=True) == "cryo"

    def test_real_ligand_survives(self):
        v = assess_ligand(Chem.MolFromSmiles(self.ESTRONE), "EST")
        assert v.ok, v.reason

    def test_artifact_rejected_by_identity_not_property(self):
        v = assess_ligand(Chem.MolFromSmiles("C[C@@H](O)CC(C)(C)O"), "MRD")
        assert not v.ok and v.reason == "artifact:cryo"

    def test_small_odorant_is_kept(self):
        """Propionate is the cognate ligand of OR51E2 (8F76).

        A conventional drug-like floor rejects it. Odorants must be volatile to
        reach a receptor, so a drug-calibrated minimum removes the target domain.
        """
        mol = Chem.MolFromSmiles(self.PROPIONATE)
        assert mol.GetNumHeavyAtoms() == 5
        assert MIN_HEAVY_ATOMS <= 5
        assert assess_ligand(mol, "PPI").ok

    def test_unlisted_polyphosphate_caught_by_property_screen(self):
        v = assess_ligand(Chem.MolFromSmiles("O=P(O)(O)OP(=O)(O)OP(=O)(O)OCC"), "ZZZ")
        assert not v.ok and v.reason == "polyphosphate"

    def test_multi_fragment_rejected(self):
        v = assess_ligand(Chem.MolFromSmiles("CCCCCCO.[Na+]"), "ZZZ")
        assert not v.ok

    def test_unparsable_is_rejected_not_raised(self):
        assert not assess_ligand(None, "ZZZ").ok


class TestChemosensoryRescue:
    """The CrossDocked blocklist rejects the on-target tastants.

    ARTIFACT_CATEGORIES was calibrated on enzymes and drug receptors, where
    glutamate and the polyamines are crystallisation additives. At a taste or
    olfactory receptor they are the stimulus: 4 codes over 9 tastepocket
    complexes, including human CaSR kokumi and fish TAAR olfaction. Nothing
    failed -- they were simply filtered out before decomposition.
    """

    def test_cognate_tastants_are_blocked_by_default(self):
        """The default must stay strict: these ARE additives in CrossDocked."""
        for code in ("GLU", "SPM", "SPD", "PUT"):
            assert is_artifact(code) == "buffer", code

    def test_cognate_tastants_are_rescued_for_chemosensory_sources(self):
        for code in ("GLU", "SPM", "SPD", "PUT"):
            assert is_artifact(code, allow_chemosensory=True) is None, code

    def test_rescue_does_not_loosen_other_categories(self):
        for code, cat in (("MRD", "cryo"), ("ADP", "cofactor"), ("HOH", "water"),
                          ("ZN", "ion"), ("NAG", "glycan")):
            assert is_artifact(code, allow_chemosensory=True) == cat, code

    def test_genuine_additives_stay_blocked(self):
        """TLA is the thaumatin crystallant -- 147 entries carry it and nothing
        else. Rescuing by category rather than by code would have admitted it."""
        for code in ("TLA", "OGA", "MLA"):
            assert is_artifact(code, allow_chemosensory=True) == "buffer", code

    def test_rescue_list_stays_narrow(self):
        """Each entry must name the structure where the code is the agonist."""
        assert set(CHEMOSENSORY_COGNATE) == {"GLU", "SPM", "SPD", "PUT"}
        for code, why in CHEMOSENSORY_COGNATE.items():
            assert any(ch.isdigit() for ch in why), f"{code}: no PDB id cited"

    def test_assess_ligand_honours_the_flag(self):
        mol = Chem.MolFromSmiles("NCCCNCCCCNCCCN")          # spermine
        assert not assess_ligand(mol, "SPM").ok
        assert assess_ligand(mol, "SPM", allow_chemosensory=True).ok


class TestFreeAminoAcidLigandsAreNotPolymerResidues:
    """A ligand named TRP is not the 14 tryptophans in the protein backbone.

    mmCIF records free amino acids as ATOM rather than HETATM, so `hetero`
    cannot find them, and selecting on `resname` alone matches every backbone
    copy too. In 7DTU that turned 2 real ligands into 30, and 445 across the
    tastepocket set -- every extra one a pocket built around a backbone residue.
    It produces pockets of the right size, full of real atoms, and errors on
    nothing. CaSR and T1R are amino-acid sensors, so this is not a rare corner.
    """

    CIF = Path.home() / "datasets/tastepocket/structures/cif/7DTU.cif"

    @pytest.fixture
    def structure(self):
        if not self.CIF.exists():
            pytest.skip(f"fixture structure not present: {self.CIF}")
        import prody

        from molplatte_prep.pocket_ligands import _parse_structure

        prody.confProDy(verbosity="none")
        return _parse_structure(self.CIF)

    def test_polymer_chains_are_identified(self, structure):
        from molplatte_prep.pocket_ligands import polymer_chain_ids

        assert set(map(str, polymer_chain_ids(structure))) == {"A", "B"}

    def test_only_the_free_tryptophans_are_ligands(self, structure):
        from molplatte_prep.pocket_ligands import ligand_instances

        inst = ligand_instances(structure, "TRP")
        assert len(inst) == 2, f"expected 2 free TRP ligands, got {len(inst)}"
        assert {str(r.getChid()) for r in inst} == {"G", "K"}
        # the free amino acid carries OXT; a backbone residue does not
        assert all(r.numAtoms() == 15 for r in inst)

    def test_naive_resname_selection_would_have_been_wrong(self, structure):
        """Pins the bug itself, so the guard cannot pass by coincidence."""
        naive = structure.select("resname TRP")
        n_naive = len(list(naive.getHierView().iterResidues()))
        assert n_naive == 30, f"fixture changed: naive select gives {n_naive}"

    def test_ordinary_hetatm_ligands_still_found(self, structure):
        """The polymer rule must not break normal HETATM ligand selection."""
        from molplatte_prep.pocket_ligands import ligand_instances

        assert len(ligand_instances(structure, "NAG")) == 14


class TestCondvecStorageDtype:
    """uint8 storage silently destroys a real-valued condition vector.

    The corpus stored rgroup_condvecs as uint8, which is correct and compact
    for 24 binary flavor bits. An ESM-2 pocket embedding is negative floats:
    -6.668 wraps to 250 and everything in (-1, 1) truncates to 0. The stored
    array then has exactly the right shape and dtype and contains none of the
    original information -- the pocket half of the corpus was integers drawn
    from {0, 250, 255} and nothing downstream could detect it.
    """

    def test_binary_encoders_stay_uint8(self):
        """Existing corpora must not be invalidated: their bits are 0/1."""
        import numpy as np

        from molplatte_prep.condvec import FlavorCondVec, NeutralCondVec

        assert np.dtype(FlavorCondVec().storage_dtype) == np.uint8
        assert np.dtype(NeutralCondVec().storage_dtype) == np.uint8

    def test_stored_pocket_declares_float(self):
        import numpy as np

        from molplatte_prep.condvec import StoredPocketCondVec

        assert np.dtype(StoredPocketCondVec(8).storage_dtype) == np.float32

    def test_two_part_widens_to_the_real_valued_half(self):
        """24 binary bits beside 1280 floats must not be stored as uint8."""
        import numpy as np

        from molplatte_prep.condvec import (FlavorCondVec, PocketCondVec,
                                            StoredPocketCondVec, TwoPartCondVec)

        zeros = TwoPartCondVec(FlavorCondVec(), PocketCondVec(dim=0))
        assert np.dtype(zeros.storage_dtype) == np.uint8

        stored = TwoPartCondVec(FlavorCondVec(), StoredPocketCondVec(1280))
        assert np.dtype(stored.storage_dtype) == np.float32

    def test_lossy_integer_cast_raises_instead_of_wrapping(self):
        """A negative float must not become 250."""
        import numpy as np

        import preprocess_flavor as pf

        pf._W["condvec"] = type("E", (), {"storage_dtype": np.uint8})()
        with pytest.raises(ValueError, match="does not survive the cast"):
            pf._store_condvecs([np.array([[-6.668, 0.3, 1.0]], dtype=np.float32)])

    def test_float_storage_preserves_negative_values(self):
        import numpy as np

        import preprocess_flavor as pf

        pf._W["condvec"] = type("E", (), {"storage_dtype": np.float32})()
        v = np.array([[-6.668, 0.318, 0.0]], dtype=np.float32)
        out = pf._store_condvecs([v])
        assert out.dtype == np.float32
        assert np.allclose(out, v)


class TestReadersRunToExhaustion:
    """Every reader must survive reaching the end of its stream.

    A crossdocked logging block sat at the end of read_coconut, referring to a
    `dropped` counter that only read_crossdocked defines. read_coconut raised
    NameError the moment its loop finished -- so any corpus build using coconut
    died partway with a traceback from inside the worker pool. It went unnoticed
    because the corpora on disk predate the CrossDocked reader, and every later
    build that touched coconut had not been re-run until now.

    Exhaustion is the untested path in a generator: everything before the last
    yield works fine.
    """

    @pytest.fixture(scope="class")
    def sources(self):
        from molplatte_prep.readers import SOURCES

        return SOURCES

    @pytest.mark.parametrize("name", ["flavordb", "coconut"])
    def test_reader_survives_the_end_of_its_stream(self, sources, name):
        from molplatte_prep.readers import SizeFilter, read_source

        path = Path(sources[name][1])
        if not path.exists():
            pytest.skip(f"{name} source not present at {path}")
        # `limit` exits through the same tail as natural exhaustion.
        n = sum(1 for _ in read_source(name, str(path), limit=25,
                                       size_filter=SizeFilter()))
        assert n > 0, f"{name} yielded nothing"

    def test_crossdocked_owns_its_dropped_counter(self):
        """The counter and the line that logs it must be in the same function."""
        import inspect

        from molplatte_prep import readers

        xd = inspect.getsource(readers.read_crossdocked)
        co = inspect.getsource(readers.read_coconut)
        assert "dropped" in xd, "read_crossdocked should build the counter"
        assert "dropped" not in co, (
            "read_coconut references a counter it does not define; "
            "the logging block belongs to read_crossdocked")


class TestAssemblyRoundTrip:
    """Detaching an R-group and reattaching it must return the parent molecule.

    This is the ground-truth control for the assembly path: it uses the STORED
    joint chemistry rather than a prediction, so any loss here is the plumbing
    itself -- feature restoration, linker-id pairing, edge reindexing -- and not
    the head. Measured 6/6 exact when this was written.

    Without it, an assembly bug is invisible: attach_rgroups returns a graph of
    the right size either way, and a wrong bond order or a dropped hydrogen
    still yields a molecule that sanitises.
    """

    SMILES = ["COc1cc(C=O)ccc1O", "CC(=O)OC1=CC=CC=C1C(=O)O",
              "CC(C)=CCCC(C)=CC=O", "CCCCCC=O", "c1ccc(cc1)C=O",
              "CC(=O)Nc1ccc(O)cc1"]

    def test_detach_then_reattach_is_identity(self):
        from molplatte_prep.decompose import decompose_molecule, wash
        from molplatte_prep.graph_ops import attach_rgroups
        from molplatte_prep.mol_features import mol_to_pyg, pyg_to_mol
        from molplatte_prep.molpla_instance import build_instance

        KW = dict(method="naveja_recap", ratio=1.0 / 3.0, include_ring=True,
                  max_cores=4, min_rgroup_atoms=2)
        checked = 0
        for smi in self.SMILES:
            mol = wash(smi, remove_stereo=False, neutralise=False)
            _, decs = decompose_molecule(mol, do_wash=False, **KW)
            if not decs:
                continue
            dec = decs[0]
            k = len(dec.rgroups)
            inst = build_instance(mol_to_pyg(mol), dec, [i != 0 for i in range(k)],
                                  mol_id="rt", store_orig=True, compute_hashes=False)
            lid = int(inst.joint_linker_ids[0])
            R = inst.R[0] if isinstance(inst.R, (list, tuple)) else inst.R
            merged = attach_rgroups(inst.P, [R], linker_ids=[lid],
                                    restore_features=True)
            got = Chem.MolToSmiles(pyg_to_mol(merged, sanitize=True))
            assert got == Chem.MolToSmiles(mol), f"{smi}: round-trip gave {got}"
            checked += 1
        assert checked >= 5, f"only {checked} molecules decomposed; fixture too weak"


class TestZincTrancheSampling:
    """ZINC's tranches are chemistry, not just directories.

    The first letter of a tranche is a molecular-weight bin (B~233 Da to
    J~475 Da), the second a logP bin. ZINC's natural distribution puts 52% of
    its mass in the D and E rows at 315-339 Da and only 2.8% in the B row -- the
    row closest to flavour chemistry. A proportional draw therefore pretrains on
    a narrow drug-like band while looking like a large diverse sample.
    """

    def test_equal_split_hits_the_target_exactly(self):
        from molplatte_prep.readers import _zinc_quotas

        caps = {"BIG1": 50_000_000, "BIG2": 50_000_000, "SMALL": 1_000, "TINY": 10}
        q = _zinc_quotas(caps, 1_000_000)
        assert sum(q.values()) == 1_000_000, "shortfall was not redistributed"

    def test_no_tranche_is_drawn_beyond_capacity(self):
        from molplatte_prep.readers import _zinc_quotas

        caps = {"A": 500, "B": 10_000_000, "C": 12}
        q = _zinc_quotas(caps, 100_000)
        for t, n in q.items():
            assert n <= caps[t], f"{t}: asked {n} of {caps[t]} available"

    def test_terminates_when_supply_is_short(self):
        """Demanding more than exists must return what exists, not loop."""
        from molplatte_prep.readers import _zinc_quotas

        q = _zinc_quotas({"A": 100, "B": 100}, 1_000_000)
        assert sum(q.values()) == 200


class TestCorpusIdsAreUnique:
    """A corpus manifest indexes decompositions BY POSITION in `ids`.

    Records are written one file per mol_id, so a repeated id means the later
    write silently overwrote the earlier one -- while the manifest kept BOTH
    entries, each with its own decomposition count. The dataset then indexes
    decomposition k of a molecule that may have fewer than k.

    Observed on a 10M ZINC draw before read_zinc deduplicated ids: 485,573
    repeated ids, of which 23.5% recorded conflicting decomposition counts, and
    9,873,383 claimed records against 9,325,378 files actually on disk. Nothing
    raised: the dataset built, and eight probe items loaded fine.
    """

    CORPORA = Path.home() / "preprocessed" / "molplatte"

    def _corpora(self):
        return sorted(p for p in self.CORPORA.glob("*/naveja_recap/__meta__.json"))

    def test_no_corpus_repeats_a_mol_id(self):
        import json

        found = self._corpora()
        if not found:
            pytest.skip("no built corpora on this machine")
        for meta_path in found:
            meta = json.loads(meta_path.read_text())
            ids = meta["ids"]
            dupes = len(ids) - len(set(ids))
            assert dupes == 0, (
                f"{meta_path.parent.parent.name}: {dupes:,} repeated ids -- "
                "later writes overwrote earlier ones and the manifest still "
                "indexes both")

    def test_manifest_count_matches_the_id_list(self):
        import json

        found = self._corpora()
        if not found:
            pytest.skip("no built corpora on this machine")
        for meta_path in found:
            meta = json.loads(meta_path.read_text())
            assert meta["n_records"] == len(meta["ids"]), (
                f"{meta_path.parent.parent.name}: n_records "
                f"{meta['n_records']:,} != {len(meta['ids']):,} ids")


class TestOdorlessIsNotFabricatedForZinc:
    """`odorless` is a volatility claim and must not be invented from MW.

    The MW>350 rule is meaningful for food compounds and natural products, where
    "is this an odorant?" is the operative question. On a virtual screening
    library it is not: nobody assessed volatility, so asserting `odorless`
    manufactures a measurement. Measured before the flag existed: 87% of a ZINC
    sample was labelled odorless on molecular weight alone.
    """

    def test_infinite_cutoff_yields_unknown_not_odorless(self):
        heavy = {"inchikey": "NOT-IN-ANY-TABLE", "mol_id": "", "mw": 480.0}
        assert bits(FlavorCondVec().encode(None, heavy)) == {"odorless"}
        suppressed = FlavorCondVec(mw_cutoff=float("inf"))
        assert bits(suppressed.encode(None, heavy)) == {"unknown"}

    def test_suppression_does_not_touch_real_labels(self):
        enc = FlavorCondVec(measured={"K": ["sweet"]}, mw_cutoff=float("inf"))
        assert bits(enc.encode(None, {"inchikey": "K", "mw": 900.0})) == {"sweet"}


class TestRetrievalGalleryIsCapped:
    """FAISSRetrieval searches the validation gallery AGAINST ITSELF.

    Cost is quadratic in the validation set, and the callback carried a comment
    justifying a CPU index with "the gallery here is thousands of rows, not
    millions". That held for a 393K-molecule corpus, whose 5% split is ~20K
    items. On ZINC the split is 1,509,773 -- 2.3 TRILLION pairs. The run wedged
    inside a single search for three hours, emitting no output and no error;
    only a stack dump showed where it was.

    An assumption stated in a comment is not a guard. This is the guard.
    """

    def test_oversized_gallery_is_subsampled(self):
        import sys

        pytest.importorskip("pytorch_lightning",
                            reason="training stack absent; FAISSRetrieval needs it")

        sys.path.insert(0, str(TRAIN_SRC))
        import numpy as np
        import torch

        from callbacks.FAISSRetrieval import FAISSRetrieval

        class FakePL:
            device = "cpu"

            def log_dict(self, *a, **k): pass

            def log(self, *a, **k): pass

        N, cap, D = 3000, 400, 16
        cb = FAISSRetrieval(max_gallery=cap)
        rng = np.random.default_rng(0)
        cb._q = [torch.from_numpy(rng.standard_normal((N, D)).astype("float32"))]
        cb._g = [torch.from_numpy(rng.standard_normal((N, D)).astype("float32"))]
        cb._keys = [f"h{i % 50}" for i in range(N)]
        cb._compute_and_log(FakePL(), stage="val")   # must not raise or hang
        assert len(cb._keys) == cap, (
            f"keys were not subsampled with q/g: {len(cb._keys)} != {cap}; "
            "gid is built from _keys and must line up row-for-row")

    def test_default_cap_is_set(self):
        import sys

        pytest.importorskip("pytorch_lightning",
                            reason="training stack absent; FAISSRetrieval needs it")

        sys.path.insert(0, str(TRAIN_SRC))
        from callbacks.FAISSRetrieval import FAISSRetrieval

        assert FAISSRetrieval().max_gallery > 0, "uncapped gallery is quadratic"


# ---------------------------------------------------------------- hashing
class TestHashVersionPinned:
    def test_hash_version_matches_built_corpora(self):
        """Corpora on disk store rgroup_hashes at this version.

        A bump without a rebuild makes every retrieval target miss, and the
        symptom is a quiet drop in Hit@K rather than an error.
        """
        from molplatte_prep.graph_hash import HASH_VERSION

        assert HASH_VERSION == 4
