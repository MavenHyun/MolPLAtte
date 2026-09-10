"""AutoDock Vina scoring for generated compounds.

The model's own `retrieval_score` says an R-group fits the LEARNED distribution
of what belongs at a joint. It says nothing about whether the resulting molecule
fits the actual pocket in 3D. Docking is the first independent check, and for
pocket-conditioned runs it is the one that matters: it can disagree with the
model, which is the whole point of adding it.

A docking setup fails SILENTLY -- a wrong receptor protonation, a mis-centred
box, a receptor that still contains its crystal ligand all return a number
rather than an error. So `redock_control` is a first-class function here, not a
footnote: it re-docks the ligand that came out of the crystal and reports RMSD
to its own pose. Under ~2 A means the setup reproduces known truth. Run it once
per receptor before trusting any score from it.

Positive Vina energies are not a bug. They mean the ligand cannot be placed
without clashing, which is a real answer -- vanillin scores +3.4 against
OR51E2 (8F76), whose pocket is sized for propionate.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["VinaDocker", "DockingUnavailable"]


class DockingUnavailable(RuntimeError):
    """Vina, meeko or obabel is missing, or receptor preparation failed."""


def _require():
    try:
        import meeko  # noqa: F401
        import vina  # noqa: F401
    except ImportError as exc:  # noqa: BLE001
        raise DockingUnavailable(
            "docking needs the `vina` and `meeko` packages "
            "(pip install vina meeko)"
        ) from exc
    if not _which("obabel"):
        raise DockingUnavailable(
            "docking needs Open Babel on PATH for receptor preparation"
        )


def _which(name: str) -> Optional[str]:
    from shutil import which

    return which(name)


def _ligand_pdbqt(smiles: str, seed: int = 0xC0FFEE) -> Optional[str]:
    """SMILES -> 3D, minimised, PDBQT string. None if any step fails."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        # A single embedding attempt fails on strained or crowded molecules;
        # random-coordinate initialisation usually recovers them.
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
    except Exception:  # noqa: BLE001 - unminimised is still dockable
        pass
    try:
        setups = MoleculePreparation().prepare(mol)
        text, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
        return text if ok else None
    except Exception:  # noqa: BLE001
        return None


@dataclass
class VinaDocker:
    """A prepared receptor plus its box, reusable across many ligands.

    `compute_vina_maps` dominates setup cost, so it runs once here and every
    subsequent dock reuses the maps. Construct one per (receptor, site).
    """

    receptor_pdbqt: Path
    center: Tuple[float, float, float]
    box_size: Tuple[float, float, float]
    exhaustiveness: int = 8
    _vina: object = None
    _xtal: Optional[np.ndarray] = None
    _pocket_atoms: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- build
    @classmethod
    def from_structure(cls, structure: str | Path, *,
                       ligand_resname: Optional[str] = None,
                       pad: float = 10.0, min_box: float = 18.0,
                       exhaustiveness: int = 8,
                       workdir: Optional[Path] = None) -> "VinaDocker":
        """Prepare a receptor from a .cif/.pdb and box it on its crystal ligand.

        The crystal ligand defines the site AND is removed from the receptor --
        leaving it in would have every docked pose clash with it.
        """
        _require()
        import prody

        import sys
        prep = Path(__file__).resolve().parents[2] / "molplatte_preprocess" / "src"
        if str(prep) not in sys.path:
            sys.path.insert(0, str(prep))
        from molplatte_prep.pocket_ligands import _parse_structure

        structure = Path(structure)
        st = _parse_structure(structure)
        if st is None:
            raise DockingUnavailable(f"could not parse {structure}")

        het = st.select("not protein and not water and not hydrogen")
        if ligand_resname:
            lig = st.select(f"resname {ligand_resname}")
        elif het is not None:
            # Largest hetero residue = the cognate ligand, not an ion or buffer.
            best, lig = -1, None
            for rn in set(het.getResnames()):
                sel = st.select(f"resname {rn}")
                if sel is not None and sel.numAtoms() > best:
                    best, lig = sel.numAtoms(), sel
        else:
            lig = None
        if lig is None or lig.numAtoms() < 3:
            raise DockingUnavailable(
                f"{structure.name} has no hetero ligand to centre a box on; "
                "pass ligand_resname, or centre the box explicitly"
            )

        work = Path(workdir or tempfile.mkdtemp(prefix="molplatte_dock_"))
        work.mkdir(parents=True, exist_ok=True)
        rec_pdb, rec_qt = work / "receptor.pdb", work / "receptor.pdbqt"
        prody.writePDB(str(rec_pdb), st.select("protein and not hydrogen"))
        cmd = ["obabel", str(rec_pdb), "-O", str(rec_qt), "-xr",
               "-p", "7.4", "--partialcharge", "gasteiger"]
        subprocess.run(cmd, capture_output=True, check=False)
        if not rec_qt.is_file() or rec_qt.stat().st_size == 0:
            raise DockingUnavailable(f"obabel produced no receptor for {structure.name}")

        xtal = lig.getCoords()
        centre = xtal.mean(0)
        size = np.maximum(xtal.max(0) - xtal.min(0) + pad, min_box)
        obj = cls(receptor_pdbqt=rec_qt,
                  center=tuple(map(float, centre)),
                  box_size=tuple(map(float, size)),
                  exhaustiveness=exhaustiveness)
        obj._xtal = xtal
        # Receptor heavy atoms near the site, for drawing the pose in context.
        try:
            near = st.select(f"protein and not hydrogen and within 12 of somepoint",
                             somepoint=lig)
            obj._pocket_atoms = near.getCoords() if near is not None else None
        except Exception:  # noqa: BLE001
            obj._pocket_atoms = None
        return obj

    def _engine(self):
        if self._vina is None:
            from vina import Vina

            v = Vina(sf_name="vina", verbosity=0)
            v.set_receptor(str(self.receptor_pdbqt))
            v.compute_vina_maps(center=list(self.center),
                                box_size=list(self.box_size))
            self._vina = v
        return self._vina

    # ----------------------------------------------------------------- use
    def dock(self, smiles: str, n_poses: int = 5,
             pose_out: Optional[Path] = None) -> Optional[float]:
        """Best binding affinity in kcal/mol, or None if the ligand failed.

        With `pose_out`, the top pose is also written there as SDF -- a docking
        score without its pose cannot be inspected, and "why is this ranked
        first" is usually answered by looking at the pose, not the number.
        """
        text = _ligand_pdbqt(smiles)
        if text is None:
            return None
        try:
            v = self._engine()
            v.set_ligand_from_string(text)
            v.dock(exhaustiveness=self.exhaustiveness, n_poses=n_poses)
            energy = float(v.energies(n_poses=1)[0][0])
        except Exception as exc:  # noqa: BLE001 - a failed dock is a None, not a crash
            logger.debug("dock failed for %s: %s", smiles, exc)
            return None
        if pose_out is not None:
            try:
                self._write_pose(v, Path(pose_out), smiles, energy)
            except Exception as exc:  # noqa: BLE001 - never lose a score over a file
                logger.debug("pose export failed for %s: %s", smiles, exc)
        return energy

    @staticmethod
    def _write_pose(v, out: Path, smiles: str, energy: float) -> None:
        from meeko import PDBQTMolecule, RDKitMolCreate
        from rdkit import Chem

        rd = RDKitMolCreate.from_pdbqt_mol(
            PDBQTMolecule(v.poses(n_poses=1), is_dlg=False, skip_typing=True))[0]
        rd.SetProp("_Name", smiles)
        rd.SetProp("vina_score", f"{energy:.3f}")
        out.parent.mkdir(parents=True, exist_ok=True)
        w = Chem.SDWriter(str(out))
        w.write(rd)
        w.close()

    def dock_many(self, smiles: Sequence[str],
                  pose_dir: Optional[Path] = None) -> dict:
        """Dock each unique SMILES; optionally write every pose into pose_dir."""
        out = {}
        for i, s in enumerate(smiles):
            if not s or s in out:
                continue
            pose = (Path(pose_dir) / f"pose_{i:03d}.sdf") if pose_dir else None
            out[s] = self.dock(s, pose_out=pose)
        return out

    def redock_control(self, ligand_smiles: str) -> Optional[float]:
        """RMSD between the redocked crystal ligand and its own pose.

        THE check that the setup is sound. Under ~2 A means the receptor prep,
        protonation and box reproduce a known answer; above it, every score
        from this receptor is suspect no matter how reasonable it looks.
        """
        if self._xtal is None:
            return None
        from meeko import PDBQTMolecule, RDKitMolCreate

        text = _ligand_pdbqt(ligand_smiles)
        if text is None:
            return None
        try:
            v = self._engine()
            v.set_ligand_from_string(text)
            v.dock(exhaustiveness=max(self.exhaustiveness, 16), n_poses=10)
            rd = RDKitMolCreate.from_pdbqt_mol(
                PDBQTMolecule(v.poses(n_poses=1), is_dlg=False, skip_typing=True))[0]
            pose = rd.GetConformer().GetPositions()
            heavy = [a.GetIdx() for a in rd.GetAtoms() if a.GetAtomicNum() > 1]
            P = pose[heavy]
            # Nearest-neighbour RMSD: symmetry-agnostic, and the atom ORDER of a
            # PDBQT round trip does not match the crystal residue's.
            d = np.linalg.norm(P[:, None, :] - self._xtal[None, :, :], axis=2)
            return float(np.sqrt((d.min(1) ** 2).mean()))
        except Exception as exc:  # noqa: BLE001
            logger.debug("redock control failed: %s", exc)
            return None
