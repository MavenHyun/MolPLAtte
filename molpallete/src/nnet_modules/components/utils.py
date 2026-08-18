"""Embedding-vocab sizes for the per-atom / per-bond categorical
features produced by ``data_modules.mol_features``.

Each value is the count of valid indices for that feature (i.e.
``len(RDKIT_FEATURES[name])``).  The mask token is reserved at index
``FEAT2DIM[name]`` (one past the last valid index), so when building
an ``nn.Embedding`` use::

    nn.Embedding(FEAT2DIM_NODE[name] + 1, hidden_dim)

so the mask slot is allocated too.

``atomic_attrdist`` is intentionally absent — it is a continuous
3-vector (normalised counts of ``core / rgroup / masked`` roles
across decompositions), not a categorical index.
"""

FEAT2DIM_NODE = {
    "atomic_num":      128,
    "formal_charge":    11,
    "chiral_tag":        9,
    "hybridization":     9,
    "num_explicit_hs":   9,
    "is_aromatic":       2,
}

FEAT2DIM_EDGE = {
    "bond_type":        22,
    "edge_is_aromatic":  2,
    "is_conjugated":     2,
    "bond_dir":          7,
    "bond_stereo":       8,
}


# ---------------------------------------------------------------------------
# torch_geometric.nn Conv factory — pick the right constructor by name.
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch_geometric import nn as pygnn

from .custom_convs import EdgeGatedGraphConv


# Convs that ignore edge_attr entirely (call site must NOT pass edge_attr=).
# Selecting one of these silently discards all five bond attributes, which
# also starves the assembly head's bond-recovery task. Kept for ablations only.
EDGE_BLIND_CONVS = {"GCNConv", "SAGEConv", "GraphConv", "ChebConv", "GATConv"}

# PyG convs that cannot carry per-edge *features*, mapped to the MolPallete
# adaptation that can. GatedGraphConv scales messages by a scalar edge_weight;
# EdgeGatedGraphConv gates them elementwise by the bond embedding instead,
# keeping the GRU node update. Point the user at the adapter rather than
# silently discarding bond information.
_NO_EDGE_ATTR_SUPPORT = {
    "GatedGraphConv": ("takes a scalar edge_weight, not an edge_attr vector",
                       "EdgeGatedGraphConv"),
}

# MolPallete-local convs, looked up before falling through to torch_geometric.nn.
_LOCAL_CONVS = {"EdgeGatedGraphConv": EdgeGatedGraphConv}

# Node-degree histogram over the ZINC-1pct anchored corpus (core + rgroups,
# 10,240 molecules): index = degree, value = node count. Molecular graphs cap
# out at degree 4. PNAConv needs this for its degree scalers.
MOL_DEGREE_HISTOGRAM = [0, 59374, 123543, 63908, 3478]


def _make_conv(name: str, in_dim: int, out_dim: int, edge_dim: int) -> nn.Module:
    if name in _NO_EDGE_ATTR_SUPPORT:
        why, alternative = _NO_EDGE_ATTR_SUPPORT[name]
        raise ValueError(
            f"{name!r} cannot be used as a MolPallete backbone as-is: it {why}. "
            f"MolPallete encodes five bond attributes per edge. Use "
            f"{alternative!r} instead -- it keeps {name}'s GRU node update but "
            f"gates each message by the bond embedding.")

    if name in _LOCAL_CONVS:
        return _LOCAL_CONVS[name](out_dim, num_layers=1, edge_dim=edge_dim)

    Cls = getattr(pygnn, name)
    mlp = lambda i, o: nn.Sequential(nn.Linear(i, o), nn.PReLU(), nn.Linear(o, o))

    if name in EDGE_BLIND_CONVS:
        return Cls(in_dim, out_dim)
    if name in {"GATv2Conv", "TransformerConv", "ResGatedGraphConv", "GENConv"}:
        return Cls(in_dim, out_dim, edge_dim=edge_dim)
    if name == "PDNConv":
        # hidden_channels sizes the internal edge->weight MLP; it has no default.
        return Cls(in_dim, out_dim, edge_dim=edge_dim, hidden_channels=edge_dim)
    if name == "PNAConv":
        return Cls(in_dim, out_dim, edge_dim=edge_dim,
                   aggregators=["mean", "min", "max", "std"],
                   scalers=["identity", "amplification", "attenuation"],
                   deg=torch.tensor(MOL_DEGREE_HISTOGRAM))
    if name == "CGConv":
        return Cls(channels=in_dim, dim=edge_dim)
    if name == "GINEConv":
        return Cls(nn=mlp(in_dim, out_dim), edge_dim=edge_dim)
    if name == "NNConv":
        # NNConv's edge network emits a full in_dim x out_dim weight matrix per
        # edge. Routing that through mlp(edge_dim, in_dim*out_dim) would build a
        # Linear(in*out, in*out) -- 1.36e9 params per layer at hidden=192. Use a
        # single projection through a narrow bottleneck instead.
        bottleneck = max(16, edge_dim // 8)
        return Cls(in_dim, out_dim, nn=nn.Sequential(
            nn.Linear(edge_dim, bottleneck), nn.PReLU(),
            nn.Linear(bottleneck, in_dim * out_dim)))
    if name in {"GMMConv", "SplineConv"}:
        return Cls(in_dim, out_dim, dim=edge_dim, kernel_size=3)
    raise ValueError(f"Unsupported GNN Conv: {name!r}")


# ---------------------------------------------------------------------------
# Norm factory — pick a normalisation layer by string name.
# ---------------------------------------------------------------------------

# Per-graph PyG norms that take the batch-assignment vector in forward,
# i.e. ``mod(x, batch)``. Plain ``BatchNorm`` and ``MessageNorm`` ignore it.
BATCH_AWARE_NORM_NAMES = {"GraphNorm", "LayerNorm", "InstanceNorm",
                          "DiffGroupNorm", "GraphSizeNorm"}

# Mapping from the name the user puts in YAML → the plain ``torch.nn``
# fallback used when no graph context (``batch`` index) is available,
# e.g. inside the per-node ``fusion_node`` MLP.
_PLAIN_NORM_FALLBACK = {
    "GraphNorm":    nn.BatchNorm1d,
    "BatchNorm":    nn.BatchNorm1d,
    "InstanceNorm": nn.InstanceNorm1d,
    "LayerNorm":    nn.LayerNorm,
}


def _make_norm(name, dim: int, *, graph_aware: bool = True) -> nn.Module:
    """Return a normalisation layer.

    ``graph_aware=True`` returns the ``torch_geometric.nn`` variant
    (which accepts ``forward(x, batch)`` for per-graph stats).
    ``graph_aware=False`` returns a pointwise ``torch.nn`` fallback for
    use inside plain MLPs that have no ``batch`` index.

    ``name`` may be ``None`` / ``"None"`` / ``"Identity"`` to disable.
    """
    if name is None or name in ("None", "Identity"):
        return nn.Identity()
    if graph_aware:
        Cls = getattr(pygnn, name, None)
        if Cls is None:
            raise ValueError(
                f"Unknown norm: {name!r} (not found in torch_geometric.nn). "
                f"Try one of: GraphNorm, LayerNorm, BatchNorm, InstanceNorm, "
                f"PairNorm, MessageNorm.")
        return Cls(dim)
    Cls = _PLAIN_NORM_FALLBACK.get(name)
    if Cls is None:
        raise ValueError(
            f"No pointwise fallback for norm {name!r}. Supported: "
            f"{sorted(_PLAIN_NORM_FALLBACK)}")
    return Cls(dim)


# ---------------------------------------------------------------------------
# GraphSequential — like nn.Sequential, but routes (x, edge_index, edge_attr)
# through PyG message-passing layers while letting pointwise modules
# (activations, dropouts) operate on x only, and batch-aware PyG norms
# (GraphNorm/LayerNorm/InstanceNorm) get the ``batch`` index.
# ---------------------------------------------------------------------------

from torch_geometric.nn import MessagePassing


class GraphSequential(nn.Module):
    """Chain a heterogeneous list of modules around a PyG graph.

    Per-step dispatch:
      * ``MessagePassing`` whose class name is in ``EDGE_BLIND_CONVS``
        → ``x = mod(x, edge_index)``
      * any other ``MessagePassing``
        → ``x = mod(x, edge_index, edge_attr=edge_attr)``
      * batch-aware PyG norms (``GraphNorm``, ``LayerNorm``,
        ``InstanceNorm``, …) → ``x = mod(x, batch)``
      * everything else (``nn.PReLU``, ``nn.Dropout``, ``nn.BatchNorm1d``,
        ``nn.Linear``, …) → ``x = mod(x)``

    **Residual connections.** Each ``MessagePassing`` layer opens a block; the
    block runs until the next ``MessagePassing`` layer (or the end of the
    chain), and the input to the block is added back to its output:

        x_{i+1} = x_i + dropout(act(norm(conv(x_i))))

    This adds no parameters and does not change any ``state_dict`` key -- but
    it *does* change the numerics, so a checkpoint trained without residuals
    will load cleanly and then behave differently. Every conv here is built
    hidden_dim -> hidden_dim, so the shapes always line up.
    """

    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(modules)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        residual = None
        for mod in self.layers:
            if isinstance(mod, MessagePassing):
                if residual is not None:
                    x = x + residual        # close the previous block
                residual = x                # open this one
                if type(mod).__name__ in EDGE_BLIND_CONVS:
                    x = mod(x, edge_index)
                else:
                    x = mod(x, edge_index, edge_attr=edge_attr)
            elif type(mod).__name__ in BATCH_AWARE_NORM_NAMES:
                x = mod(x, batch)
            else:
                x = mod(x)
        if residual is not None:
            x = x + residual                # close the final block
        return x
