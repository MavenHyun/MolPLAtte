"""Extract and validate drug-like ligands from pocket-ligand structures.

Structural datasets are full of molecules that are in the crystal but are not
binders: cryoprotectants, buffer components, ions, detergents used to solubilise
membrane proteins, and the nucleotide cofactors that co-purify with half the
enzymes in the PDB. In CrossDocked2020 they dominate the head of the
distribution -- the twelve most frequent ligands are ADP, SAH, MRD, OGA, ATP,
NAD, GDP, UMP, ANP, APC, AMP and PPV, together roughly 8,700 pocket pairs.

Filtering them by physicochemical property does not work. Measured on the
CrossDocked head: S-adenosylhomocysteine scores QED 0.35 and benzamidine 0.46,
while estrone -- a real ligand -- scores 0.78 and staurosporine 0.30. A QED
threshold that removes the artifacts also removes genuine binders.

What does work is identity. Every ligand carries a PDB chemical component
dictionary (CCD) code, and artifacts are a known, enumerable set. So the primary
filter here is a curated CCD blocklist, and the property checks are a secondary
screen for things the blocklist has not seen.

Two deliberate design points:

- **Cofactors are blocked by default but separable.** ATP in a kinase is a real
  cognate ligand, not an artifact. `is_artifact` reports the CATEGORY, so a
  caller that wants cofactors can keep them; the default excludes them because
  they bind almost everything and would dominate any R-group vocabulary.
- **The blocklist is explicitly a curated subset, not exhaustive.** It covers
  the categories that actually appear at high frequency. Anything unlisted falls
  through to the property screen rather than being silently trusted.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

__all__ = [
    "ARTIFACT_CATEGORIES",
    "ARTIFACT_CODES",
    "COFACTOR_CODES",
    "LigandVerdict",
    "ccd_code_from_path",
    "is_artifact",
    "assess_ligand",
    "extract_ligands",
    "pocket_residues",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# CCD blocklist
# --------------------------------------------------------------------------
#: Grouped so callers can re-admit a category instead of editing a flat set.
#: Codes are PDB chemical component identifiers, upper case.
ARTIFACT_CATEGORIES: Dict[str, Set[str]] = {
    "water": {"HOH", "DOD", "WAT", "H2O"},
    # monatomic and simple inorganic ions
    "ion": {
        "NA", "K", "LI", "RB", "CS", "MG", "CA", "SR", "BA", "MN", "FE", "FE2",
        "CO", "3CO", "NI", "CU", "CU1", "CU3", "ZN", "CD", "HG", "AU", "AG",
        "PT", "PD", "TL", "PB", "AL", "GA", "IN", "SB", "AS", "SE", "MO", "W",
        "V", "CR", "Y", "YB", "GD", "EU", "SM", "LU", "HO", "TB", "CE", "LA",
        "CL", "BR", "IOD", "F", "FLO", "NO3", "NO2", "SO4", "SO3", "PO4", "PI",
        "AZI", "CN", "SCN", "OH", "O", "OXY", "PER", "BF4", "PF6",
    },
    # cryoprotectants and precipitants
    "cryo": {
        "EDO", "GOL", "MRD", "MPD", "PGO", "PGE", "PG4", "P6G", "1PE", "2PE",
        "7PE", "12P", "15P", "XPE", "PE3", "PE4", "PE5", "PE8", "DMS", "TFA",
        "SUC", "TRE", "MAL", "SOR", "XYL", "GLC", "BET", "IPA", "MOH", "EOH",
        "ACN", "ACT", "ACY", "FMT", "PEG",
    },
    # buffers and common additives
    "buffer": {
        "TRS", "EPE", "MES", "BIS", "CIT", "FLC", "TLA", "TAR", "MLA", "MLI",
        "MAE", "SIN", "GLU", "IMD", "BCT", "CO3", "CAC", "PPV", "POP", "DPO",
        "OGA", "BEN", "BAM", "GAI", "URE", "NH4", "SPD", "SPM", "PUT", "MYR",
        "HED", "BME", "DTT", "DTU", "TCE", "PYR", "OXL", "MAN",
    },
    # sugars and N-/O-glycans
    "glycan": {
        "NAG", "NDG", "NGA", "A2G", "BMA", "MAN", "BGC", "GAL", "GLA", "FUC",
        "FUL", "XYS", "XYP", "SIA", "NAN", "NGC", "RAM", "RIB", "ARA", "LMT",
        "GLP", "G6P", "F6P",
    },
    # lipids, sterols and detergents (membrane-protein crystallography)
    "lipid": {
        "OLA", "OLB", "OLC", "PLM", "STE", "MYS", "PEE", "PC1", "PCW", "PGV",
        "PSC", "LHG", "DGA", "CLR", "CHD", "CHS", "Y01", "HC3", "D10", "D12",
        "LDA", "LMU", "BOG", "BNG", "C8E", "OGA1", "UNL", "UND", "HEX", "DAO",
        "DDQ", "TRD", "MC3", "9PE", "3PE", "PEF", "SQD",
    },
    # nucleotides and enzyme cofactors -- real binders, but promiscuous
    "cofactor": {
        "ATP", "ADP", "AMP", "ANP", "ACP", "APC", "AGS", "ADX", "A12",
        "GTP", "GDP", "GMP", "GNP", "GSP", "G2P", "GCP",
        "UTP", "UDP", "UMP", "UPG", "U5P",
        "CTP", "CDP", "CMP", "C5P",
        "TTP", "TDP", "TMP", "THP",
        "IMP", "ITP", "XMP",
        "NAD", "NAI", "NAP", "NDP", "NAX", "NHD",
        "FAD", "FMN", "FDA", "RBF",
        "COA", "ACO", "COO", "SCA", "MCA",
        "SAH", "SAM", "MTA", "MET",
        "TPP", "TDP1", "PLP", "PMP", "P5P",
        "BTN", "THF", "FOL", "MTX1", "B12", "COB", "CNC",
        "HEM", "HEC", "HEA", "HEB", "SRM", "DHE", "VER",
        "PQQ", "MGD", "MOS", "F43", "H4B", "BH4",
        "GSH", "GDS", "GTT",
    },
}

#: Cofactors are separable so a caller can re-admit them deliberately.
COFACTOR_CODES: Set[str] = set(ARTIFACT_CATEGORIES["cofactor"])

#: Flat blocklist, every category.
ARTIFACT_CODES: Set[str] = {c for s in ARTIFACT_CATEGORIES.values() for c in s}

#: `..._rec_<pdb>_<ccd>_lig_...` -- CrossDocked encodes the ligand's CCD code in
#: its filename, which is how a set with no CCD column can still be filtered.
_CCD_IN_PATH = re.compile(r"_rec_[0-9a-z]{4}_([0-9a-z]{1,3})_lig", re.I)


def ccd_code_from_path(path: str) -> Optional[str]:
    """CCD code embedded in a CrossDocked ligand filename, or ``None``."""
    m = _CCD_IN_PATH.search(str(path))
    return m.group(1).upper() if m else None


def is_artifact(code: Optional[str], *, keep_cofactors: bool = False) -> Optional[str]:
    """Artifact category of *code*, or ``None`` if it is not blocklisted.

    ``keep_cofactors`` re-admits nucleotides and enzyme cofactors, which are
    genuine cognate ligands for the enzymes that use them.
    """
    if not code:
        return None
    code = code.strip().upper()
    for name, codes in ARTIFACT_CATEGORIES.items():
        if code in codes:
            if name == "cofactor" and keep_cofactors:
                return None
            return name
    return None


# --------------------------------------------------------------------------
# drug-likeness
# --------------------------------------------------------------------------
#: Elements a small-molecule ligand may contain. Anything else is a metal
#: cluster, a heavy-atom derivative used for phasing, or a modified residue.
_ALLOWED_ELEMENTS = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H", "B", "Se"}

#: Matches the corpus SizeFilter, NOT a drug-likeness convention. Standard
#: drug-like minimums (>=10 heavy atoms, MW>=150) are calibrated on drugs and
#: are actively wrong here: odorants must be volatile to reach a receptor, so
#: they are small by necessity. Propionate -- 5 heavy atoms -- is the cognate
#: ligand of OR51E2 in 8F76, the only genuine human olfactory receptor structure
#: in the PDB. A drug-like floor would discard exactly the chemistry this
#: project exists to model.
MIN_HEAVY_ATOMS = 5
MAX_HEAVY_ATOMS = 70
MAX_MW = 900.0
#: Phosphate-rich molecules at this count are nucleotides or sugar phosphates
#: rather than drug-like binders, even when the CCD code is unfamiliar.
MAX_PHOSPHORUS = 2


@dataclass
class LigandVerdict:
    """Why a ligand was kept or rejected. Never a bare boolean -- the counts
    per reason are what tell you whether a filter is working or overreaching."""

    ok: bool
    reason: str = ""
    code: Optional[str] = None
    heavy_atoms: int = 0
    mw: float = 0.0
    smiles: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.ok


def assess_ligand(
    mol: Optional[Chem.Mol],
    code: Optional[str] = None,
    *,
    keep_cofactors: bool = False,
    min_heavy_atoms: int = MIN_HEAVY_ATOMS,
    max_heavy_atoms: int = MAX_HEAVY_ATOMS,
) -> LigandVerdict:
    """Decide whether *mol* is a plausible drug-like ligand.

    Identity first (CCD blocklist), then structure. No QED / Lipinski gate: on
    the CrossDocked head those scores do not separate artifacts from binders
    (SAH 0.35, benzamidine 0.46, estrone 0.78), so a threshold tuned to remove
    the former removes plenty of the latter.
    """
    cat = is_artifact(code, keep_cofactors=keep_cofactors)
    if cat:
        return LigandVerdict(False, f"artifact:{cat}", code)
    if mol is None:
        return LigandVerdict(False, "unparsable", code)

    n = mol.GetNumHeavyAtoms()
    smi = Chem.MolToSmiles(mol)
    mw = Descriptors.MolWt(mol)
    v = LigandVerdict(True, "", code, n, mw, smi)

    if n < min_heavy_atoms:
        return LigandVerdict(False, "too_small", code, n, mw, smi)
    if n > max_heavy_atoms:
        return LigandVerdict(False, "too_large", code, n, mw, smi)
    if mw > MAX_MW:
        return LigandVerdict(False, "mw_too_high", code, n, mw, smi)

    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    bad = elements - _ALLOWED_ELEMENTS
    if bad:
        return LigandVerdict(False, f"element:{','.join(sorted(bad))}", code, n, mw, smi)
    if sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "P") > MAX_PHOSPHORUS:
        return LigandVerdict(False, "polyphosphate", code, n, mw, smi)
    # A ligand deposited as several disconnected pieces is a salt or a mixture;
    # the corpus keeps one molecule per record.
    if len(Chem.GetMolFrags(mol)) > 1:
        return LigandVerdict(False, "multi_fragment", code, n, mw, smi)
    if not any(a.GetSymbol() == "C" for a in mol.GetAtoms()):
        return LigandVerdict(False, "inorganic", code, n, mw, smi)
    return v


# --------------------------------------------------------------------------
# structure parsing (ProDy)
# --------------------------------------------------------------------------
def _parse_structure(path: Path):
    """Parse a .cif/.pdb into a ProDy AtomGroup."""
    import prody

    prody.confProDy(verbosity="none")
    suffix = path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        return prody.parseMMCIF(str(path))
    return prody.parsePDB(str(path))


def extract_ligands(
    path: str | Path,
    *,
    keep_cofactors: bool = False,
    min_heavy_atoms: int = MIN_HEAVY_ATOMS,
) -> List[Tuple[str, LigandVerdict, object]]:
    """Every non-water heteroatom residue in *path*, with a verdict each.

    Returns ``(residue_key, verdict, prody_selection)``. The selection is kept
    so a caller can compute the pocket around a ligand it decided to keep,
    without reparsing the structure.

    Bond orders are NOT in the coordinate file. They are inferred by RDKit from
    geometry, which is unreliable for the aromatic and charged groups that
    matter most here -- so the SMILES on the verdict is a fallback. Prefer the
    CCD ideal SDF (``structures/ligands/<CODE>_ideal.sdf``) when one exists;
    that carries deposited bond orders.
    """
    path = Path(path)
    try:
        st = _parse_structure(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[pocket_ligands] cannot parse %s: %s", path.name, exc)
        return []
    if st is None:
        return []
    het = st.select("hetero and not water")
    if het is None:
        return []

    out: List[Tuple[str, LigandVerdict, object]] = []
    for res in het.getHierView().iterResidues():
        code = res.getResname().strip().upper()
        key = f"{path.stem}_{res.getChid()}_{code}_{int(res.getResnum())}"
        sel = st.select(
            "chain {} and resname {} and resnum {}".format(
                res.getChid(), code, int(res.getResnum())
            )
        )
        mol = _selection_to_mol(sel)
        verdict = assess_ligand(
            mol, code, keep_cofactors=keep_cofactors, min_heavy_atoms=min_heavy_atoms
        )
        out.append((key, verdict, sel))
    return out


def _selection_to_mol(sel) -> Optional[Chem.Mol]:
    """ProDy selection -> RDKit mol, with bond orders inferred from geometry."""
    if sel is None or sel.numAtoms() == 0:
        return None
    try:
        import io

        import prody

        buf = io.StringIO()
        prody.writePDBStream(buf, sel)
        mol = Chem.MolFromPDBBlock(buf.getvalue(), removeHs=True, sanitize=True)
        if mol is None:
            mol = Chem.MolFromPDBBlock(buf.getvalue(), removeHs=True, sanitize=False)
        return mol
    except Exception:  # noqa: BLE001
        return None


def pocket_residues(structure, ligand_sel, cutoff: float = 10.0):
    """Protein residues with any atom within *cutoff* angstroms of the ligand.

    10 A matches the ``pocket10`` convention CrossDocked's processed LMDB uses,
    so pockets extracted here are directly comparable to the ones already in it.
    """
    if ligand_sel is None:
        return None
    return structure.select(f"protein and within {cutoff} of lig", lig=ligand_sel)
