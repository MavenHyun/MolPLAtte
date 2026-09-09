"""Lead optimization at inference: any flavor compound, any protein structure.

Given a molecule and (optionally) a pocket, decompose the molecule into a core
plus R-group slots and retrieve replacement R-groups for each slot from the
library, ranked by the trained co-embedding.

The whole point of MolPLAtte is that the *condition* is exogenous. A query is
built from three things that a user actually has:

    core template   the molecule they want to modify, minus one substituent
    flavor profile  the sensory profile they are aiming for
    pocket          the receptor they want it to bind

None of the three is derived from the answer. That constraint is what the
project's earlier condvec failed -- a fragment vector computed from the intact
molecule contains the R-group being retrieved, and zeroing it at inference took
hit@1 from 0.6111 to exactly 0.0000.

The retrieval path deliberately reuses ``build_instance`` and
``collate_molplatte`` rather than reimplementing the marker arithmetic. The
callback that reports Hit@K during training reads ``batch["query_projection"]``
from the same forward pass, so a number produced here and a number reported
there cannot drift apart.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem

from data_modules.base import collate_molplatte
from data_modules.dataset import MolPLAtteSample
from nnet_modules.molplatte import MolPLAtte

logger = logging.getLogger(__name__)

PREP_SRC = Path(__file__).resolve().parents[2] / "molplatte_preprocess" / "src"
if str(PREP_SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PREP_SRC))

from molplatte_prep.condvec import FLAVOR_LABELS  # noqa: E402
from molplatte_prep.decompose import decompose_molecule, wash  # noqa: E402
from molplatte_prep.mol_features import mol_to_pyg  # noqa: E402
from molplatte_prep.graph_ops import attach_rgroups  # noqa: E402
from molplatte_prep.mol_features import pyg_to_mol  # noqa: E402
from molplatte_prep.molpla_instance import build_instance  # noqa: E402
from data_modules.rgroup_vocab import RGroupLibraryVocab  # noqa: E402

#: Must match the corpus. A different decomposition yields different WL hashes,
#: so retrieval targets stop matching the library and Hit@K drops quietly
#: instead of raising.
DECOMP_KWARGS = dict(
    method="naveja_recap",
    ratio=1.0 / 3.0,
    include_ring=True,
    max_cores=4,
    min_rgroup_atoms=2,
)


def _n_aromatic(smiles: str) -> int:
    m = Chem.MolFromSmiles(smiles) if smiles else None
    return sum(1 for a in m.GetAtoms() if a.GetIsAromatic()) if m else -1


@dataclass
class Suggestion:
    """One retrieved R-group for one slot of one core."""

    rank: int
    score: float
    smiles: str
    hash: str
    corpus_count: int
    is_novel: bool
    #: SMILES of the molecule this R-group builds when attached to the core.
    #: ``None`` when assembly was not requested, the checkpoint has no assembly
    #: head, or the predicted joint chemistry does not sanitise.
    product: Optional[str] = None
    #: Why ``product`` is None, when it is.
    product_error: str = ""
    #: False when the product sanitises but lost aromatic atoms the parent had.
    #: Reported separately because RDKit accepts a broken aromatic ring without
    #: complaint -- "sanitises" and "chemically right" are different claims, and
    #: an undertrained assembly head fails this while scoring 100% valid.
    aromaticity_kept: Optional[bool] = None

    def __repr__(self) -> str:  # pragma: no cover - display only
        tag = " [novel]" if self.is_novel else ""
        got = f"  -> {self.product}" if self.product else ""
        if self.product and self.aromaticity_kept is False:
            got += "  [aromaticity broken]"
        return f"#{self.rank} {self.smiles}  score={self.score:.4f}{tag}{got}"


@dataclass
class SlotResult:
    """The slot that was emptied, and what the model would put back."""

    decomp_index: int
    slot_index: int
    core_smiles: str
    original_rgroup: str
    suggestions: List[Suggestion]


class LeadOptimizer:
    """Checkpoint + R-group library, loaded once, queried many times."""

    def __init__(self, model: MolPLAtte, vocab, device: str = "cuda",
                 popularity_coef: float = 1.0,
                 temperature: float = 0.1) -> None:
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = device
        self.popularity_coef = float(popularity_coef)
        # MUST match the R-group contrastive loss temperature (contrastive.py
        # default 0.1). The corrected score is sim/tau + coef * log p(k), and
        # tau sets the scale at which the two terms trade off. Omitting it --
        # scoring sim + log p directly, as this did until 2026-09-09 -- leaves
        # the similarity term ~10x too small to matter: measured on the shipped
        # checkpoint, the model spanned 0.083 across candidates while log p
        # spanned 1.6, so `optimize()` returned the frequency prior in corpus-
        # count order no matter what flavor or pocket was supplied. Retrieval
        # eval showed 13x lift over that same prior, so the signal was there
        # and simply never reached inference.
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)
        self._library: Optional[torch.Tensor] = None
        self._log_prior: Optional[torch.Tensor] = None
        self._assembly_untrained = False
        prov = getattr(vocab, "provenance", None) or {}
        self._novel = set(prov.get("novel_hashes") or [])

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, checkpoint: str | Path, vocab_path: str | Path,
             device: str = "cuda", popularity_coef: float = 1.0,
             temperature: float = 0.1, **model_kwargs) -> "LeadOptimizer":
        """Load a checkpoint and its library.

        ``model_kwargs`` must reproduce the architecture the checkpoint was
        trained with -- above all ``condvec_dim`` and ``pocket_input_dim``. The
        query projector's input width is a function of both, so a mismatch is a
        shape error at load rather than a silently wrong model.
        """
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        model = MolPLAtte(**model_kwargs)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"checkpoint has {len(unexpected)} unexpected tensors, "
                             f"e.g. {unexpected[:3]}; model_kwargs do not match it")
        if missing:
            logger.warning("[LeadOptimizer] %d tensors missing from the checkpoint "
                           "and left at init: %s", len(missing), missing[:3])
        # An assembly head at random init produces confident nonsense -- observed
        # emitting hafnium atoms into aromatic rings -- so record that it is
        # untrained and refuse to use it, rather than logging a warning nobody
        # reads and returning molecules anyway.
        untrained = any("assembly_head" in k for k in missing)
        # The SAME wrapper the training-time retrieval callback uses, so a
        # number produced here and a Hit@K reported during validation are
        # scored against identical row order and identical priors.
        vocab = RGroupLibraryVocab(vocab_path)
        opt = cls(model, vocab, device=device,
                  popularity_coef=popularity_coef, temperature=temperature)
        opt._assembly_untrained = untrained
        if untrained:
            logger.warning("[LeadOptimizer] the checkpoint has NO assembly-head "
                           "weights; assembly is disabled. Retrain with "
                           "assembly.enabled=true to build molecules.")
        return opt

    # -- library -----------------------------------------------------------
    @torch.no_grad()
    def build_library(self, batch_size: int = 1024) -> None:
        """Embed every R-group in the library with the CURRENT weights.

        The library is only valid for the weights that built it -- both
        projectors are trained, so an index built by an earlier checkpoint
        scores against a different space and produces plausible nonsense.
        """
        chunks = []
        for graph_batch in self.vocab.batches(batch_size):
            graph_batch = graph_batch.to(self.device)
            chunks.append(self.model.encode_rgroups(graph_batch).float().cpu())
        library = torch.cat(chunks, dim=0)
        self._library = torch.nn.functional.normalize(library, dim=-1)

        prior = np.asarray(self.vocab.frequency_prior, dtype=np.float64)
        self._log_prior = torch.from_numpy(
            np.log(np.clip(prior, 1e-12, None)).astype(np.float32)
        )
        logger.info("[LeadOptimizer] library: %s rows", f"{len(library):,}")

    # -- condition vector --------------------------------------------------
    def flavor_vector(self, labels: Sequence[str]) -> np.ndarray:
        """24 flavor bits from label names.

        Never all-zero: an empty request becomes ``unknown``, which is a real
        bit the model was trained on. Zeros would mean "no flavor at all",
        a claim the caller did not make.
        """
        vec = np.zeros(len(FLAVOR_LABELS), dtype=np.float32)
        index = {l: i for i, l in enumerate(FLAVOR_LABELS)}
        unknown = []
        for label in labels or ():
            i = index.get(str(label).strip().lower())
            if i is None:
                unknown.append(label)
            else:
                vec[i] = 1.0
        if unknown:
            raise ValueError(f"unknown flavor labels {unknown}; "
                             f"choose from {list(FLAVOR_LABELS)}")
        if not vec.any():
            vec[index["unknown"]] = 1.0
        return vec

    def condvec(self, flavor: Optional[Sequence[str]] = None,
                pocket: Optional[np.ndarray] = None) -> np.ndarray:
        """Assemble ``[flavor | pocket]`` at the width the model expects."""
        want = self.model.config.condvec_dim
        if want == 0:
            return np.zeros(0, dtype=np.float32)
        flavor_vec = self.flavor_vector(flavor or ())
        pocket_width = self.model.config.pocket_input_dim
        if not pocket_width:
            return flavor_vec
        if pocket is None:
            # Zeros are the honest encoding of "no pocket supplied":
            # PocketConditioning masks an all-zero pocket half to exactly zero,
            # so the query falls back to flavor alone rather than to noise.
            pocket_vec = np.zeros(pocket_width, dtype=np.float32)
        else:
            pocket_vec = np.asarray(pocket, dtype=np.float32).ravel()
            if pocket_vec.shape[0] != pocket_width:
                raise ValueError(f"pocket embedding has width {pocket_vec.shape[0]}, "
                                 f"model expects {pocket_width}")
        return np.concatenate([flavor_vec, pocket_vec]).astype(np.float32)

    # -- retrieval ---------------------------------------------------------
    @torch.no_grad()
    def optimize(self, smiles: str, *, flavor: Optional[Sequence[str]] = None,
                 pocket: Optional[np.ndarray] = None, top_k: int = 10,
                 max_decompositions: int = 4,
                 assemble: bool = False) -> List[SlotResult]:
        """Retrieve replacement R-groups for every slot of *smiles*.

        ``assemble=True`` additionally builds each suggested molecule, which
        needs a checkpoint trained with ``assembly.enabled=true``. Without one
        every suggestion comes back with ``product=None`` and a stated reason
        rather than silently unassembled.
        """
        if self._library is None:
            self.build_library()

        mol = wash(smiles, remove_stereo=False, neutralise=False)
        if mol is None:
            raise ValueError(f"could not parse {smiles!r}")
        mol_data = mol_to_pyg(mol)
        # returns (washed_mol, decompositions); do_wash=False because the
        # molecule is already washed above and washing twice can change it.
        _, decomps = decompose_molecule(mol, do_wash=False, **DECOMP_KWARGS)
        if not decomps:
            raise ValueError(f"{smiles!r} did not decompose under "
                             f"{DECOMP_KWARGS['method']}")

        cv = self.condvec(flavor, pocket)
        results: List[SlotResult] = []

        for d_idx, decomp in enumerate(decomps[:max_decompositions]):
            n = len(decomp.rgroups)
            for slot in range(n):
                # Detach exactly one R-group; the rest stay attached. That is
                # the lead-optimization question -- replace THIS substituent --
                # rather than the training-time multi-detach.
                islinked = [i != slot for i in range(n)]
                try:
                    instance = build_instance(
                        mol_data, decomp, islinked,
                        mol_id="query", decomp_idx=d_idx,
                        store_orig=False, compute_hashes=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("slot %d of decomp %d failed: %s", slot, d_idx, exc)
                    continue

                n_detached = len(instance.detached_indices)
                sample = MolPLAtteSample(
                    G=instance.G, P=instance.P, R=instance.R,
                    condvec=torch.from_numpy(np.tile(cv, (n_detached, 1))),
                    R_hashes=[""] * n_detached,
                    joint_linker_ids=instance.joint_linker_ids,
                    joint_G_atoms=instance.joint_G_atoms,
                    joint_metas={}, islinked=instance.islinked,
                    mol_id="query", instance_id=instance.instance_id, smiles=smiles,
                )
                batch = collate_molplatte([sample])
                batch = {k: (v.to(self.device) if torch.is_tensor(v)
                             else v.to(self.device) if hasattr(v, "to") else v)
                         for k, v in batch.items()}
                out = self.model(batch)
                query = out["query_projection"].float().cpu()
                query = torch.nn.functional.normalize(query, dim=-1)

                scores = (query @ self._library.T) / self.temperature
                if self.popularity_coef:
                    # InfoNCE optimises PMI, log p(k|q) - log p(k), not the
                    # posterior. Adding log p(k) back is what makes the ranking
                    # comparable to a frequency prior instead of systematically
                    # favouring rare R-groups.
                    scores = scores + self.popularity_coef * self._log_prior
                top = scores[0].topk(min(top_k, scores.shape[1]))

                sugg = [
                    Suggestion(
                        rank=r + 1,
                        score=float(sc),
                        smiles=self._smiles_of_row(int(i)),
                        hash=self.vocab.hashes[int(i)],
                        corpus_count=int(self.vocab.counts[int(i)])
                        if hasattr(self.vocab, "counts") else 0,
                        is_novel=self.vocab.hashes[int(i)] in self._novel,
                    )
                    for r, (sc, i) in enumerate(zip(top.values, top.indices))
                ]

                if assemble:
                    # The template is instance.P: the core with this joint
                    # masked. joint_linker_ids is 1:1 with the detached
                    # R-groups, and exactly one is detached here.
                    lid = int(instance.joint_linker_ids[0])
                    for r, (_sc, i) in enumerate(zip(top.values, top.indices)):
                        graph = self._graph_of_row(int(i), lid)
                        if graph is None:
                            sugg[r].product_error = "library row has no graph"
                            continue
                        prod, err = self.assemble(instance.P, graph, lid)
                        sugg[r].product = prod
                        sugg[r].product_error = err
                        if prod:
                            sugg[r].aromaticity_kept = (
                                _n_aromatic(prod) >= _n_aromatic(smiles))

                results.append(SlotResult(
                    decomp_index=d_idx,
                    slot_index=slot,
                    core_smiles=getattr(decomp, "core_smiles", "") or "",
                    original_rgroup=self._rgroup_smiles_of(mol, decomp, slot),
                    suggestions=sugg,
                ))
        return results

    # -- assembly ----------------------------------------------------------
    @property
    def has_assembly_head(self) -> bool:
        """True only when the head exists AND carries trained weights."""
        return ("assembly_head" in self.model.nnet
                and not self._assembly_untrained)

    @torch.no_grad()
    def assemble(self, template, rgroup_graph, linker_id: int) -> tuple:
        """Attach *rgroup_graph* to *template*, predicting the joint chemistry.

        Retrieval says WHICH R-group belongs at a joint. It cannot say HOW to
        attach it: the joint is masked, so the shared linker atom's identity and
        the bond that reforms were both replaced by MASK sentinels. That masking
        is deliberate -- it frees retrieval from having to match the original
        linker -- but it means a retrieved R-group cannot be bonded on without
        the assembly head predicting what the mask destroyed.

        The template's stored ``linker_metas`` hold the ORIGINAL joint, which is
        the wrong answer for any R-group other than the one that was there. So
        the predicted attributes are written over them.

        Returns ``(smiles, error)``; exactly one is non-empty. The four
        attributes are predicted by independent classifiers and can disagree --
        a carbon with charge +2 and five hydrogens is representable in the
        feature space and not in chemistry -- so a sanitisation failure is an
        ordinary outcome, reported rather than raised.
        """
        if "assembly_head" not in self.model.nnet:
            return None, "model was built without an assembly head"
        if self._assembly_untrained:
            return None, ("assembly head is at random init -- this checkpoint "
                          "was trained with assembly.enabled=false")

        import copy

        from torch_geometric.data import Batch

        head = self.model.nnet["assembly_head"]
        enc = self.model.nnet["graph_encoder"]

        tpl = copy.deepcopy(template)
        rg = copy.deepcopy(rgroup_graph)
        try:
            core_at = int((tpl.linker_id == linker_id).nonzero()[0].item())
            rg_at = int((rg.linker_id == linker_id).nonzero()[0].item()) \
                if hasattr(rg, "linker_id") and (rg.linker_id == linker_id).any() \
                else int(rg.is_linker.nonzero()[0].item())
        except (IndexError, AttributeError) as exc:
            return None, f"no joint {linker_id} on template or R-group: {exc}"

        # One encoder pass over both views, exactly as training does.
        batch = Batch.from_data_list([tpl, rg]).to(self.device)
        out = enc(batch)
        H = out.node_embeddings
        offset = int((batch.batch == 0).sum().item())
        fused = head._fuse(H[core_at].unsqueeze(0), H[offset + rg_at].unsqueeze(0))

        atom_pred = {a: int(h(fused).argmax(-1).item()) for a, h in head.node_heads.items()}
        bond_pred = {a: int(h(fused).argmax(-1).item()) for a, h in head.edge_heads.items()}

        metas = dict(getattr(tpl, "linker_metas", {}) or {})
        meta = dict(metas.get(linker_id) or {})
        atom_feats = dict(meta.get("atom_features") or {})
        atom_feats.update(atom_pred)          # predicted attrs win over the original
        bond_feats = dict(meta.get("cut_bond_features") or {})
        bond_feats.update(bond_pred)
        meta["atom_features"] = atom_feats
        meta["cut_bond_features"] = bond_feats
        metas[linker_id] = meta
        tpl.linker_metas = metas

        try:
            merged = attach_rgroups(tpl, [rg], linker_ids=[linker_id],
                                    restore_features=True,
                                    bond_features={linker_id: bond_feats})
        except Exception as exc:  # noqa: BLE001
            return None, f"attach failed: {type(exc).__name__}: {exc}"
        try:
            mol = pyg_to_mol(merged, sanitize=True)
            smi = Chem.MolToSmiles(mol)
        except Exception as exc:  # noqa: BLE001
            return None, f"unsanitisable: {type(exc).__name__}"
        return (smi, "") if smi else (None, "empty SMILES")

    # -- display helpers ---------------------------------------------------
    def _graph_of_row(self, row: int, linker_id: int):
        """Library R-group graph, with its linker relabelled to this joint.

        Vocabulary graphs are stored with whatever linker_id they carried in
        their source molecule. attach_rgroups pairs template joint to R-group
        clone BY id, so a mismatched id silently finds no partner.
        """
        import copy

        graphs = getattr(self.vocab, "_graphs", None)
        if not graphs or row >= len(graphs) or graphs[row] is None:
            return None
        g = copy.deepcopy(graphs[row])
        if not hasattr(g, "is_linker"):
            return None
        idx = g.is_linker.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return None
        if not hasattr(g, "linker_id") or g.linker_id is None:
            g.linker_id = torch.zeros(g.num_nodes, dtype=torch.long)
        g.linker_id = g.linker_id.clone()
        g.linker_id[:] = 0
        g.linker_id[idx[0]] = linker_id
        return g

    def _smiles_of_row(self, row: int) -> str:
        smiles = getattr(self.vocab, "smiles", None)
        if smiles and row < len(smiles) and smiles[row]:
            return smiles[row]
        return "?"

    @staticmethod
    def _rgroup_smiles_of(mol, decomp, slot: int) -> str:
        try:
            atoms = list(decomp.rgroups[slot].rgroup_atoms)
            return Chem.MolFragmentToSmiles(mol, atomsToUse=atoms)
        except Exception:  # noqa: BLE001
            return "?"


# ---------------------------------------------------------------- pockets
def pocket_embedding_from_structure(
    path: str | Path,
    *,
    ligand_code: Optional[str] = None,
    cutoff: float = 10.0,
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    device: str = "cuda",
) -> np.ndarray:
    """ESM-2 pocket embedding for a ``.pdb`` / ``.cif``, matching the corpus.

    Same recipe as ``embed_pockets_esm.py``: residues within *cutoff* of the
    ligand, mean-pooled over every pocket residue of every chain. Anything else
    -- a different cutoff, per-chain pooling, a different checkpoint -- puts the
    query in a different space from the library and retrieval degrades without
    reporting anything.

    ``ligand_code`` picks which heteroatom residue defines the pocket. Left
    ``None``, the largest valid ligand in the structure is used.
    """
    import prody

    from molplatte_prep.pocket_ligands import (_parse_structure, ligand_instances,
                                               pocket_residues)

    prody.confProDy(verbosity="none")
    path = Path(path)
    structure = _parse_structure(path)
    if structure is None:
        raise ValueError(f"could not parse {path}")

    if ligand_code:
        candidates = ligand_instances(structure, ligand_code.upper())
        if not candidates:
            raise ValueError(f"no ligand {ligand_code!r} in {path.name}")
    else:
        het = structure.select("hetero and not water")
        if het is None:
            raise ValueError(f"no heteroatom ligand in {path.name}; pass ligand_code")
        candidates = sorted(het.getHierView().iterResidues(),
                            key=lambda r: r.numAtoms(), reverse=True)
    residue = candidates[0]

    sel = structure.select(
        f"chain {residue.getChid()} and resname {residue.getResname()} "
        f"and resnum {int(residue.getResnum())}"
    )
    pocket = pocket_residues(structure, sel, cutoff=cutoff)
    if pocket is None:
        raise ValueError(f"no protein within {cutoff} A of the ligand in {path.name}")

    import sys as _sys

    _sys.path.insert(0, str(PREP_SRC.parent / "scripts"))
    from embed_pockets_esm import verify_offset
    from molplatte_prep.pocket_ligands import chain_sequence
    from transformers import AutoModel, AutoTokenizer

    residues = [(r.getChid(), int(r.getResnum())) for r in pocket.getHierView().iterResidues()]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    n_prefix = verify_offset(tokenizer)
    esm = AutoModel.from_pretrained(model_name).to(device).eval()

    rows = []
    for chid in sorted({c for c, _ in residues}):
        seq, index_of = chain_sequence(structure, chid)
        idx = sorted(index_of[num] for c, num in residues if c == chid and num in index_of)
        if not seq or not idx or len(seq) < 20:
            continue
        enc = tokenizer([seq], return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = esm(**enc).last_hidden_state[0].float().cpu().numpy()
        emb = hidden[n_prefix:n_prefix + len(seq)]
        rows.append(emb[[i for i in idx if i < len(emb)]])

    if not rows:
        raise ValueError(f"no mappable pocket chain in {path.name}")
    return np.concatenate(rows, axis=0).mean(axis=0).astype(np.float32)
