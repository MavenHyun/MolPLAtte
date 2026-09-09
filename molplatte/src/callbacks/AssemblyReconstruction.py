"""End-to-end molecule reconstruction, logged every validation epoch.

The three losses say whether embeddings are separating. They cannot say whether
the model can BUILD a molecule, which is what lead optimization actually
delivers. Assembly is the step that turns "which R-group" into an attachable
answer, and it fails in ways a loss curve does not show:

* per-attribute cross-entropy can look excellent while the joint is wrong,
  because the attributes are independent classifiers and only their AGREEMENT
  produces a valid atom;
* a molecule can sanitise and still be wrong -- RDKit accepts a broken aromatic
  ring without complaint, so "it parsed" and "it is chemically right" are
  different claims and are logged as different metrics;
* chirality is not intrinsic. CHI_TETRAHEDRAL_CW/CCW is defined against bond
  ORDER, so a reattached joint can carry the correct tag and denote the mirror
  image. That was a real defect here (21 of 22 failures, connectivity intact),
  and for flavour work it matters -- carvone's enantiomers smell of spearmint
  and caraway.

So this reconstructs real molecules and compares canonical SMILES.

Every epoch it also logs the GROUND-TRUTH control: the same path with the stored
joint chemistry instead of predictions. Without it, a drop is ambiguous between
"the head got worse" and "the graph plumbing broke", and those have completely
different fixes.

Logged under ``{stage}/assembly/``:

    exact              rebuilt SMILES == parent SMILES
    exact_control      the same, using stored chemistry (plumbing ceiling)
    head_gap           control - exact, i.e. the head's own contribution
    aromaticity_kept   product retains at least the parent's aromatic atoms
    sanitised          product parsed at all
    acc/<attribute>    per-attribute accuracy at the joint
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch

logger = logging.getLogger(__name__)


class AssemblyReconstruction(pl.Callback):
    """Rebuild molecules from (core, R-group) and score them.

    Parameters
    ----------
    corpus
        Directory of ``.pt`` records to draw evaluation molecules from. Defaults
        to the DataModule's own corpus.
    n_molecules
        How many to reconstruct per evaluation. Reconstruction is RDKit-bound
        and single-threaded, so this is a real cost: ~400 molecules is a few
        seconds, 10,000 is minutes. Kept small and stated rather than silently
        subsampled.
    every_n_epochs, enable_after_epoch
        Reconstruction on an untrained head measures nothing, and the metric
        starting mid-run is clearer than a flat line of zeros.
    """

    def __init__(self,
                 corpus: Optional[str] = None,
                 n_molecules: int = 300,
                 every_n_epochs: int = 1,
                 enable_after_epoch: int = 1,
                 seed: int = 20260909):
        super().__init__()
        self.corpus = Path(corpus) if corpus else None
        self.n_molecules = int(n_molecules)
        self.every_n_epochs = int(every_n_epochs)
        self.enable_after_epoch = int(enable_after_epoch)
        self.seed = int(seed)
        self._smiles: Optional[List[str]] = None
        self._disabled = False

    # ------------------------------------------------------------------ #
    def _load_smiles(self, trainer) -> bool:
        if self._smiles is not None:
            return bool(self._smiles)
        corpus = self.corpus
        if corpus is None:
            try:
                corpus = Path(trainer.datamodule.config.dataset_path)
            except Exception:  # noqa: BLE001
                self._disabled = True
                return False
        files = sorted(Path(corpus).rglob("*.pt"))
        if not files:
            logger.warning("[AssemblyReconstruction] no records under %s; disabled",
                           corpus)
            self._disabled = True
            return False
        random.Random(self.seed).shuffle(files)
        out: List[str] = []
        for f in files:
            if len(out) >= self.n_molecules:
                break
            try:
                d = torch.load(f, weights_only=False)
            except Exception:  # noqa: BLE001
                continue
            if d.get("smiles"):
                out.append(d["smiles"])
        self._smiles = out
        logger.info("[AssemblyReconstruction] %d molecules drawn from %s",
                    len(out), Path(corpus).parent.name)
        return bool(out)

    def _active(self, trainer) -> bool:
        if self._disabled or trainer.sanity_checking:
            return False
        if trainer.current_epoch < self.enable_after_epoch:
            return False
        return trainer.current_epoch % self.every_n_epochs == 0

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _score(self, pl_module) -> Optional[dict]:
        import copy

        from rdkit import Chem, RDLogger
        from torch_geometric.data import Batch

        RDLogger.DisableLog("rdApp.*")

        # Imported here, not at module import: the training package must stay
        # importable on a machine without the preprocessing stack.
        import sys
        prep = Path(__file__).resolve().parents[3] / "molplatte_preprocess" / "src"
        if str(prep) not in sys.path:
            sys.path.insert(0, str(prep))
        from molplatte_prep.decompose import decompose_molecule, wash
        from molplatte_prep.graph_ops import attach_rgroups
        from molplatte_prep.mol_features import mol_to_pyg, pyg_to_mol
        from molplatte_prep.molpla_instance import build_instance

        # MUST match the corpus. A different decomposition yields different
        # joints, and the reconstruction would be measuring something else.
        KW = dict(method="naveja_recap", ratio=1.0 / 3.0, include_ring=True,
                  max_cores=4, min_rgroup_atoms=2)

        model = pl_module.model.model
        # nn.ModuleDict has no .get()
        if "assembly_head" not in model.nnet:
            return None
        head = model.nnet["assembly_head"]
        was_training = model.training
        model.eval()

        n = exact = control = arom = sane = 0
        attr_ok: dict = {}
        attr_n: dict = {}
        try:
            for smi in self._smiles:
                mol = wash(smi, remove_stereo=False, neutralise=False)
                if mol is None:
                    continue
                try:
                    data = mol_to_pyg(mol)
                    _, decs = decompose_molecule(mol, do_wash=False, **KW)
                    if not decs:
                        continue
                    dec = decs[0]
                    k = len(dec.rgroups)
                    inst = build_instance(data, dec, [i != 0 for i in range(k)],
                                          mol_id="asm", store_orig=True,
                                          compute_hashes=False)
                except Exception:  # noqa: BLE001
                    continue
                lid = int(inst.joint_linker_ids[0])
                R = inst.R[0] if isinstance(inst.R, (list, tuple)) else inst.R
                parent = Chem.MolToSmiles(mol)
                n += 1

                meta = (getattr(inst.P, "linker_metas", {}) or {}).get(lid) or {}
                try:
                    tpl, rg = copy.deepcopy(inst.P), copy.deepcopy(R)
                    core_at = int((tpl.linker_id == lid).nonzero()[0].item())
                    rg_at = int(rg.is_linker.nonzero()[0].item())
                    batch = Batch.from_data_list([tpl, rg]).to(pl_module.device)
                    H = model.nnet["graph_encoder"](batch).node_embeddings
                    off = int((batch.batch == 0).sum().item())
                    fused = head._fuse(H[core_at].unsqueeze(0),
                                       H[off + rg_at].unsqueeze(0))
                    pred = {a: int(h(fused).argmax(-1).item())
                            for a, h in head.node_heads.items()}
                    pred.update({a: int(h(fused).argmax(-1).item())
                                 for a, h in head.edge_heads.items()})
                except Exception:  # noqa: BLE001
                    continue

                truth = dict(meta.get("atom_features") or {})
                truth.update(meta.get("cut_bond_features") or {})
                for a, v in pred.items():
                    if a in truth:
                        attr_n[a] = attr_n.get(a, 0) + 1
                        attr_ok[a] = attr_ok.get(a, 0) + int(v == truth[a])

                # --- rebuild with PREDICTED chemistry
                try:
                    t2 = copy.deepcopy(inst.P)
                    metas = dict(getattr(t2, "linker_metas", {}) or {})
                    mm = dict(metas.get(lid) or {})
                    af = dict(mm.get("atom_features") or {})
                    bf = dict(mm.get("cut_bond_features") or {})
                    af.update({a: v for a, v in pred.items() if a in af or a in
                               head.node_heads})
                    bf.update({a: v for a, v in pred.items() if a in head.edge_heads})
                    mm["atom_features"], mm["cut_bond_features"] = af, bf
                    metas[lid] = mm
                    t2.linker_metas = metas
                    merged = attach_rgroups(t2, [copy.deepcopy(R)],
                                            linker_ids=[lid],
                                            restore_features=True,
                                            bond_features={lid: bf})
                    got = Chem.MolToSmiles(pyg_to_mol(merged, sanitize=True))
                except Exception:  # noqa: BLE001
                    got = ""
                if got:
                    sane += 1
                    exact += int(got == parent)
                    gm = Chem.MolFromSmiles(got)
                    pm = Chem.MolFromSmiles(parent)
                    if gm is not None and pm is not None:
                        arom += int(
                            sum(1 for a in gm.GetAtoms() if a.GetIsAromatic())
                            >= sum(1 for a in pm.GetAtoms() if a.GetIsAromatic()))

                # --- control: STORED chemistry, isolates plumbing from head
                try:
                    ctl = attach_rgroups(copy.deepcopy(inst.P),
                                         [copy.deepcopy(R)], linker_ids=[lid],
                                         restore_features=True)
                    control += int(Chem.MolToSmiles(pyg_to_mol(ctl, sanitize=True))
                                   == parent)
                except Exception:  # noqa: BLE001
                    pass
        finally:
            model.train(was_training)

        if not n:
            return None
        out = {
            "exact": exact / n,
            "exact_control": control / n,
            "head_gap": (control - exact) / n,
            "aromaticity_kept": arom / n,
            "sanitised": sane / n,
            "n": float(n),
        }
        for a in attr_n:
            out[f"acc/{a}"] = attr_ok[a] / attr_n[a]
        return out

    # ------------------------------------------------------------------ #
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._active(trainer) or not self._load_smiles(trainer):
            return
        try:
            m = self._score(pl_module)
        except Exception as exc:  # noqa: BLE001 - never fail a run over a metric
            logger.warning("[AssemblyReconstruction] skipped: %s", exc)
            return
        if not m:
            return
        pl_module.log_dict({f"val/assembly/{k}": v for k, v in m.items()},
                           on_epoch=True, sync_dist=True)
        logger.info(
            "[AssemblyReconstruction] n=%d exact=%.3f (control %.3f, head gap "
            "%.3f)  aromaticity=%.3f  %s",
            int(m["n"]), m["exact"], m["exact_control"], m["head_gap"],
            m["aromaticity_kept"],
            "  ".join(f"{k.split('/')[-1]}={v:.3f}"
                      for k, v in sorted(m.items()) if k.startswith("acc/")))
