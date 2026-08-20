"""Batching for MolPLA's four views, and the Lightning DataModule.

The single-encoder-pass trick
-----------------------------
MolPLA encodes ``G``, ``P`` and every ``R`` with **one** shared encoder in **one**
forward pass: all three are concatenated into a single graph batch and split
afterwards by boolean node masks.  That is the trait that makes the framework
cheap -- one GNN call per step instead of three -- and MolPallete keeps it.

Where MolPLA hand-rolled positional marker lists, this collate emits explicit
per-graph tensors (``graph_view``, ``graph_sample``) and derives the node masks
from them.  Same cost, but the pairing between a linker atom's three incarnations
is stated as an index rather than implied by ordering.

Batch contract
--------------
``W``
    A PyG ``Batch`` over every graph in the step, ordered
    ``[G_0 .. G_{B-1}, P_0 .. P_{B-1}, R_0^0, R_0^1, .., R_{B-1}^{k-1}]``.
``graph_view`` ``long[n_graphs]``
    ``0`` = G, ``1`` = P, ``2`` = R.
``graph_sample`` ``long[n_graphs]``
    Which of the ``B`` instances each graph belongs to.
``G_markers`` / ``P_markers`` / ``R_markers`` ``bool[n_nodes]``
    Node masks for the three views.
``R_pool_index`` ``long[n_R_nodes]``
    R-group id in ``[0, J)`` per R-side node -- the pooling index that turns R
    nodes into one embedding per detached R-group.
``joint_G_idx`` / ``joint_P_idx`` / ``joint_R_idx`` ``long[J]``
    Node indices into ``W`` of the same linker atom's three incarnations: the
    intact atom in G, the masked joint in P, and the masked clone in R.  These
    are the three terms of MolPLA's linker-node objective, paired by ``linker_id``
    rather than by position.
``joint_sample`` ``long[J]``, ``joint_rgroup`` ``long[J]``
    Instance id and R-group id per joint.
``condvec`` ``float[J, condvec_dim]``
    Functional-group condition vector of each detached R-group, concatenated onto
    the query before the query projector (MolPLA Eq. 11).
``R_hashes`` ``List[str]`` (length ``J``)
    WL subgraph hash per detached R-group -- the retrieval vocabulary key and the
    multi-positive grouping key for the contrastive loss.
``mol_ids`` / ``instance_ids`` / ``smiles`` ``List[str]``

Note that ``J`` -- the number of detached R-groups in the batch -- is the batch
size of losses 2 and 3, while ``B`` is the batch size of loss 1.  MolPLA states
the same relation: ``B1 >= B2`` and ``B1 >= B3``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional

import torch
# Must be the same Lightning namespace the Trainer uses -- mixing
# `lightning` and `pytorch_lightning` makes Lightning's is_overridden()
# fail to find the parent class ("ValueError: Expected a parent").
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch

from .assembly_targets import (
    RECOVERABLE_EDGE_ATTRS,
    RECOVERABLE_NODE_ATTRS,
    derive_node_target,
)
from .dataset import MolPalleteDataset, MolPalleteSample

__all__ = ["collate_molpallete", "DataModuleConfig", "MolPalleteDataModule"]


def _linker_positions(graph) -> Dict[int, int]:
    """``linker_id -> local atom index`` for every masked joint in *graph*."""
    if not hasattr(graph, "linker_id"):
        return {}
    ids = graph.linker_id
    positions = (ids > 0).nonzero(as_tuple=False).flatten()
    return {int(ids[p]): int(p) for p in positions}


def collate_molpallete(batch: List[MolPalleteSample]) -> dict:
    """Flatten samples into the four-view batch described in the module docstring."""
    graphs: List[object] = []
    graph_view: List[int] = []
    graph_sample: List[int] = []

    # Order matters: all G first, then all P, then all R. Losses 1 slices on the
    # view masks, so keeping views contiguous makes those slices cheap.
    for i, sample in enumerate(batch):
        graphs.append(sample.G)
        graph_view.append(0)
        graph_sample.append(i)
    p_graph_offset = len(graphs)
    for i, sample in enumerate(batch):
        graphs.append(sample.P)
        graph_view.append(1)
        graph_sample.append(i)
    r_graph_offset = len(graphs)

    r_graph_of_sample: List[List[int]] = []
    for i, sample in enumerate(batch):
        local = []
        for rgroup in sample.R:
            local.append(len(graphs) - r_graph_offset)
            graphs.append(rgroup)
            graph_view.append(2)
            graph_sample.append(i)
        r_graph_of_sample.append(local)

    W = Batch.from_data_list(graphs)

    view = torch.tensor(graph_view, dtype=torch.long)
    sample_of_graph = torch.tensor(graph_sample, dtype=torch.long)
    node_view = view[W.batch]

    # `ptr` gives the first node index of each graph in W -- the offset that turns
    # a graph-local atom index into a node index into W.
    ptr = W.ptr

    joint_G, joint_P, joint_R, joint_sample, joint_rgroup = [], [], [], [], []
    condvecs, hashes = [], []
    # Pre-mask chemistry at each joint: the assembly head's targets. Populated
    # only when the dataset was asked for them, so a run without the head pays
    # nothing here.
    node_targets = {a: [] for a in RECOVERABLE_NODE_ATTRS}
    edge_targets = {a: [] for a in RECOVERABLE_EDGE_ATTRS}

    for i, sample in enumerate(batch):
        p_graph = p_graph_offset + i
        p_local = _linker_positions(sample.P)
        for j, rgroup in enumerate(sample.R):
            r_graph = r_graph_offset + r_graph_of_sample[i][j]
            linker_id = int(sample.joint_linker_ids[j])
            r_local = _linker_positions(rgroup)
            p_atom = p_local.get(linker_id)
            r_atom = r_local.get(linker_id)
            if p_atom is None or r_atom is None:
                # Every R-group carries exactly one masked clone and P carries its
                # mirror, both stamped with this linker_id, so a miss means the
                # detach produced something unpairable. Drop the joint rather than
                # emit an index that silently points at the wrong atom.
                continue

            meta = sample.joint_metas.get(linker_id) if sample.joint_metas else None
            if meta is not None:
                atom_feats = meta.get("atom_features", {})
                bond_feats = meta.get("cut_bond_features", {})
                for attr in RECOVERABLE_NODE_ATTRS:
                    node_targets[attr].append(derive_node_target(attr, atom_feats))
                for attr in RECOVERABLE_EDGE_ATTRS:
                    edge_targets[attr].append(int(bond_feats.get(attr, 0)))

            joint_G.append(int(ptr[i]) + int(sample.joint_G_atoms[j]))
            joint_P.append(int(ptr[p_graph]) + p_atom)
            joint_R.append(int(ptr[r_graph]) + r_atom)
            joint_sample.append(i)
            joint_rgroup.append(r_graph_of_sample[i][j])
            condvecs.append(sample.condvec[j])
            hashes.append(sample.R_hashes[j] if j < len(sample.R_hashes) else "")

    long = lambda xs: torch.tensor(xs, dtype=torch.long)  # noqa: E731

    # R-group id per R-side node: the graph's position among the R graphs.
    r_node_mask = node_view == 2
    graph_ids = torch.arange(len(graphs), dtype=torch.long)
    R_pool_index = (graph_ids - r_graph_offset)[W.batch][r_node_mask]

    out = {
        "W": W,
        "graph_view": view,
        "graph_sample": sample_of_graph,
        "G_markers": node_view == 0,
        "P_markers": node_view == 1,
        "R_markers": r_node_mask,
        "node_sample": sample_of_graph[W.batch],
        "R_pool_index": R_pool_index,
        "joint_G_idx": long(joint_G),
        "joint_P_idx": long(joint_P),
        "joint_R_idx": long(joint_R),
        "joint_sample": long(joint_sample),
        "joint_rgroup": long(joint_rgroup),
        "condvec": (
            torch.stack(condvecs)
            if condvecs
            else torch.zeros(0, batch[0].condvec.shape[-1] if batch else 0)
        ),
        "num_joints": len(joint_G),
        # Emitted only if every joint had metas -- a partially populated target
        # would silently train on whichever joints happened to carry them.
        "joint_atom_target": (
            {a: long(v) for a, v in node_targets.items()}
            if node_targets["atomic_num"] and len(node_targets["atomic_num"]) == len(joint_G)
            else {}
        ),
        "joint_bond_target": (
            {a: long(v) for a, v in edge_targets.items()}
            if edge_targets["bond_type"] and len(edge_targets["bond_type"]) == len(joint_G)
            else {}
        ),
        "R_hashes": hashes,
        "num_samples": len(batch),
        "mol_ids": [s.mol_id for s in batch],
        "instance_ids": [s.instance_id for s in batch],
        "smiles": [s.smiles for s in batch],
    }
    # Alias so generic callbacks and the LightningModule can ask for a graph-count
    # without knowing the view layout.
    out["G"] = W
    return out


@dataclass
class DataModuleConfig:
    """Hydra-facing configuration.

    ``__post_init__`` derives the corpus leaf the same way MolDAM does, so a run
    only has to name the method.
    """

    dataset_path: Optional[Path] = None
    dataset_version: str = "coconut-flavordb_v3"
    decomposition_method: str = "naveja_recap"
    batch_size: int = 512
    num_workers: int = 8
    val_split: float = 0.05
    test_split: float = 0.05
    seed: int = 42
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2
    condvec_mode: str = "neutral"
    condvec_dim: int = 97
    max_rgroups: int = 8
    #: Derive pre-mask joint chemistry for the assembly head. Costs a little
    #: CPU per __getitem__ and nothing on disk; off unless the head is enabled.
    need_assembly_targets: bool = False
    #: "decomposition" (MolPLA-style: every (molecule, core) pair once per epoch)
    #: or "molecule" (MolDAM-style: one core drawn per molecule per epoch).
    sampling_unit: str = "decomposition"
    #: MolPLA's common-R-group filter. `null` disables it. R-groups at or above
    #: this percentile of the corpus occurrence distribution count as "common",
    #: and an instance whose detached R-groups are more than
    #: `max_common_fraction` common is redrawn. MolPLA used 99.99 / 0.5.
    common_percentile: Optional[float] = None
    max_common_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path is required")
        if not isinstance(self.dataset_path, Path):
            self.dataset_path = Path(self.dataset_path)
        self.dataset_path = (
            self.dataset_path / self.dataset_version / self.decomposition_method
        )


class MolPalleteDataModule(pl.LightningDataModule):
    """Splits one corpus into train/val/test and serves the four-view batches."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        known = {f.name for f in fields(DataModuleConfig)}
        self.config = DataModuleConfig(**{k: v for k, v in kwargs.items() if k in known})
        self._splits: Optional[tuple] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self._splits is not None:
            return
        dataset = MolPalleteDataset(
            self.config.dataset_path,
            condvec_dim=self.config.condvec_dim,
            max_rgroups=self.config.max_rgroups,
            seed=self.config.seed,
            need_assembly_targets=self.config.need_assembly_targets,
            sampling_unit=self.config.sampling_unit,
            common_percentile=self.config.common_percentile,
            max_common_fraction=self.config.max_common_fraction,
        )
        n_total = len(dataset)
        n_val = int(n_total * self.config.val_split)
        n_test = int(n_total * self.config.test_split)
        n_train = n_total - n_val - n_test
        self._splits = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.config.seed),
        )

    def _split(self, index: int):
        if self._splits is None:
            self.setup()
        return self._splits[index]

    @property
    def train_dataset(self):
        return self._split(0)

    @property
    def val_dataset(self):
        return self._split(1)

    @property
    def test_dataset(self):
        return self._split(2)

    def _loader(self, split, shuffle: bool) -> DataLoader:
        workers = self.config.num_workers
        kwargs = dict(
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=collate_molpallete,
            pin_memory=self.config.pin_memory,
            drop_last=shuffle,
        )
        if workers > 0:
            kwargs["persistent_workers"] = self.config.persistent_workers
            kwargs["prefetch_factor"] = self.config.prefetch_factor
        return DataLoader(split, **kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
