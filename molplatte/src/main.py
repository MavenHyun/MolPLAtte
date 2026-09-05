from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict

import data_modules as dm
from data_modules import MolPLAtteDataModule       # noqa: F401  (resolved via getattr)

import nnet_modules      as nm
import loss_modules      as lossm
import lightning_modules as lm
from nnet_modules      import MolPLAtte                   # noqa: F401  (resolved via getattr)
from loss_modules      import LossModuleMolPLAtte         # noqa: F401  (resolved via getattr)
from lightning_modules import MolPLAtteLightningModule    # noqa: F401  (resolved via getattr)

from callbacks import (
    FAISSRetrieval,
    PredictionTable,
    RepresentationHealth,
    RGroupLibraryRetrieval,
    SaveBestModelCheckpoint,
)


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
torch.set_float32_matmul_precision("medium")

def get_init_config(config: DictConfig) -> DictConfig:
    if config.api.get("wandb"):
        os.environ["WANDB_API_KEY"] = config.api.wandb
    if config.api.get("huggingface"):
        os.environ["HF_TOKEN"] = config.api.huggingface

    pl.seed_everything(config.random_seed, workers=True)
    _apply_assembly_switch(config)

    base_path = Path(config.master_path)
    # Namespaced under the project, matching preprocessed/molplatte. The flat
    # ~/checkpoints was shared with MolDAM (debug_anchored_best.pt lives there),
    # so an unprefixed experiment_name could have collided across projects.
    ckpt_path = base_path / "checkpoints" / "molplatte"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    return config, base_path, ckpt_path

def _apply_assembly_switch(config: DictConfig) -> None:
    """Drive every assembly-related setting from the single ``assembly.enabled`` knob.

    The head needs three things set consistently: the nnet must build it, the
    loss must create its term, and the dataset must derive the pre-mask targets.
    Setting any subset silently produces a broken run -- most insidiously
    ``assembly_head`` without ``need_assembly_targets``, which builds the head,
    feeds it no targets, and trains on nothing while logging a plausible loss.
    One switch, propagated here, makes that unrepresentable.

    Sub-keys under ``assembly`` are still honoured, so the switch sets the
    defaults and anything explicitly overridden on the command line wins.
    """
    acfg = config.get("assembly") or {}
    OmegaConf.set_struct(config.nnet_module_kwargs, False)
    OmegaConf.set_struct(config.loss_module_kwargs, False)
    OmegaConf.set_struct(config.data_module_kwargs, False)
    if not acfg.get("enabled", False):
        # Explicitly clear, so a stale override cannot half-enable the head.
        config.nnet_module_kwargs.assembly_head = None
        config.loss_module_kwargs.loss_assembly_kwargs = None
        config.data_module_kwargs.need_assembly_targets = False
        return

    config.data_module_kwargs.need_assembly_targets = True
    config.nnet_module_kwargs.assembly_head = acfg.get("head", "AssemblyHead")
    head_kwargs = OmegaConf.to_container(acfg.get("head_kwargs") or {}, resolve=True)
    existing = OmegaConf.to_container(
        config.nnet_module_kwargs.get("assembly_head_kwargs") or {}, resolve=True
    )
    existing.update({k: v for k, v in head_kwargs.items() if v is not None})
    config.nnet_module_kwargs.assembly_head_kwargs = existing

    loss_kwargs = OmegaConf.to_container(acfg.get("loss_kwargs") or {}, resolve=True)
    config.loss_module_kwargs.loss_assembly_kwargs = loss_kwargs or {}

    weights = config.lightning_module_kwargs.get("loss_weights")
    if weights is not None and "assembly" not in weights:
        # loss_weights comes out of Hydra in struct mode, which forbids new keys.
        with open_dict(weights):
            weights["assembly"] = acfg.get("loss_weight", 0.5)

    logging.info(
        "Assembly head ENABLED  [%s, fusion=%s, coupling=%s, weight=%s]",
        config.nnet_module_kwargs.assembly_head,
        existing.get("fusion", "concat"),
        existing.get("coupling", False),
        (weights or {}).get("assembly"),
    )


def get_data_module(config: DictConfig, base_path: Path) -> MolPLAtteDataModule:
    if config.data_module_kwargs.get("dataset_path") is None:
        config.data_module_kwargs.dataset_path = str(base_path / "preprocessed" / "molplatte")

    data_module = getattr(dm, config.data_module)(**config.data_module_kwargs)
    data_module.setup()

    logging.info(f"Number of Training Samples   "
                 f"[{len(data_module.train_dataset):,}]")
    logging.info(f"Number of Validation Samples "
                 f"[{len(data_module.val_dataset):,}]")
    logging.info(f"Number of Test Samples       "
                 f"[{len(data_module.test_dataset):,}]")

    return data_module

def get_nnet_module(config: DictConfig) -> nn.Module:
    nnet_module = getattr(nm, config.nnet_module)(**config.nnet_module_kwargs)
    n_params = sum(p.numel() for p in nnet_module.parameters())
    logging.info(f"Number of Parameters         [{n_params:,}]")
    return nnet_module


def get_loss_module(config: DictConfig, nnet_module: nn.Module) -> nn.Module:
    kw = OmegaConf.to_container(config.loss_module_kwargs, resolve=True)
    return getattr(lossm, config.loss_module)(nnet_module, **kw)


def get_lightning_module(config: DictConfig, loss_module: nn.Module) -> pl.LightningModule:
    kw = OmegaConf.to_container(config.lightning_module_kwargs, resolve=True)
    return getattr(lm, config.lightning_module)(loss_module, **kw)


def get_logger(config: DictConfig, ckpt_path: Path) -> Optional[pl.loggers.Logger]:
    wcfg = config.get("wandb") or {}
    if not wcfg or wcfg.get("project") is None:
        return None
    # group / job_type / tags are what make a hyperparameter sweep navigable in
    # the wandb UI: `group` collapses every run of one sweep into a single row
    # that can be expanded, `job_type` separates screening runs from finals, and
    # tags carry the axis being varied so runs can be filtered without parsing
    # names. Without them a 20-run sweep is 20 unrelated rows.
    tags = wcfg.get("tags")
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    elif tags is not None:
        tags = [str(t) for t in tags]
    logger = pl.loggers.WandbLogger(
        project=wcfg.get("project"),
        name=wcfg.get("name"),
        group=wcfg.get("group"),
        job_type=wcfg.get("job_type"),
        tags=tags,
        notes=wcfg.get("notes"),
        save_dir=wcfg.get("save_dir") or str(ckpt_path),
        log_model=wcfg.get("log_model", False),
    )
    # Log the resolved Hydra config as hyperparameters. Without this wandb
    # records ZERO config keys -- the run page shows metrics with no record of
    # what produced them, so runs cannot be compared or reproduced from the UI.
    # api.* is stripped: it carries the resolved WANDB_API_KEY and HF_TOKEN.
    resolved = OmegaConf.to_container(config, resolve=True)
    resolved.pop("api", None)
    logger.log_hyperparams(resolved)
    return logger


def _retrieval_loss_kwargs(config: DictConfig) -> dict:
    """The R-group retrieval loss block -- MolPLA loss #3, the only one FAISS scores."""
    lmk = config.get("loss_module_kwargs") or {}
    return dict(lmk.get("loss_rgroup_kwargs") or {})


def _logq_enabled(config: DictConfig) -> bool:
    """True when the R-group retrieval loss applies the logQ correction.

    Defaults to True to match the loss module's own default. MolDAM gated this
    on ``decomposition_paradigm``; MolPLAtte has a single view scheme (MolPLA's
    G/P/R/Q views), so the branch is gone and the flag is read directly.
    """
    return bool(_retrieval_loss_kwargs(config).get("logq_correction", True))


def _retrieval_temperature(config: DictConfig) -> float:
    return float(_retrieval_loss_kwargs(config).get("temperature", 0.01))


def get_callbacks(config: DictConfig, ckpt_path: Path) -> List[pl.Callback]:
    exp_name = config.get("experiment_name")
    ckpt_filename = f"{exp_name}_best.pt" if exp_name else "molplatte_best.pt"
    es_cfg = config.get("early_stopping", {}) or {}
    es_patience = int(es_cfg.get("patience", 10))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    es_monitor  = str(es_cfg.get("monitor", "val/loss"))
    es_mode     = str(es_cfg.get("mode", "min"))
    # The final deliverable checkpoint trains on ALL records with no validation
    # split, because the fold estimates already sized the effect and holding
    # data back would only shrink the model that ships. With no val/loss to
    # monitor, EarlyStopping raises rather than degrading -- so it is switched
    # off explicitly instead of being fed a metric that does not exist.
    es_enabled = bool(es_cfg.get("enabled", True))
    ckpt_cfg = config.get("checkpoint", {}) or {}
    ckpt_monitor = str(ckpt_cfg.get("monitor", es_monitor))
    ckpt_mode    = str(ckpt_cfg.get("mode", es_mode))
    # Prediction tables land next to the rest of this run's Hydra output.
    try:
        from hydra.core.hydra_config import HydraConfig
        table_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        table_dir = os.getcwd()
    callbacks: List[pl.Callback] = [
        SaveBestModelCheckpoint(save_dir=str(ckpt_path),
                                monitor=ckpt_monitor, mode=ckpt_mode,
                                filename=ckpt_filename),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        *([pl.callbacks.EarlyStopping(monitor=es_monitor, mode=es_mode,
                                      patience=es_patience,
                                      min_delta=es_min_delta,
                                      check_finite=True,
                                      verbose=True)]
          if es_enabled else []),
        # Training and evaluation must agree on what the similarity estimates.
        # With logq_correction the critic learns log p(k|q) directly, so FAISS
        # ranks raw similarity. Without it the critic learns PMI
        # (log p(k|q) - log p(k)) and the popularity term has to be added back
        # at scoring, or r@k underreports badly -- measured on MolDAM
        # v29a/GINEConv, 0.3805 vs 0.6646. That number is MolDAM's ZINC
        # anchored corpus, not MolPLAtte's; the *mechanism* carries over
        # (skewed R-group vocabulary + in-batch InfoNCE), the magnitude does
        # not. Deriving one flag from the other prevents the mismatch either way.
        FAISSRetrieval(enable_after_epoch=3,
                       popularity_correction=not _logq_enabled(config),
                       temperature=_retrieval_temperature(config)),
        # MolDAM's MolecularReassembly callback is still absent, but the reason
        # has changed: MolPLAtte DOES have an assembly objective now
        # (assembly.enabled), so there is something to score. Reassembly is
        # scored out-of-band by evaluate_reassembly.py against a checkpoint
        # rather than per-epoch, because rebuilding molecules with RDKit on
        # every validation pass is far too slow to sit in the training loop.
        PredictionTable(k=5, max_rows=500, log_every_n_epochs=1,
                        enable_after_epoch=3, out_dir=table_dir,
                        monitor=ckpt_monitor, mode=ckpt_mode),
    ]

    # Full-corpus R-group library retrieval -- MolPLA's actual RGR evaluation.
    # FAISSRetrieval above scores against a val-split gallery, which is a much
    # easier problem; this scores against every recommendable R-group in the
    # corpus and reports each Hit@K next to the frequency-prior baseline.
    # Disables itself with a warning if no vocabulary has been built.
    library_cfg = config.get("rgroup_library") or {}
    if library_cfg.get("enabled", True):
        callbacks.append(RGroupLibraryRetrieval(
            vocab_path=library_cfg.get("vocab_path"),
            max_entries=library_cfg.get("max_entries"),
            every_n_epochs=int(library_cfg.get("every_n_epochs", 1)),
            enable_after_epoch=int(library_cfg.get("enable_after_epoch", 2)),
            search_k=int(library_cfg.get("search_k", 1000)),
            encode_batch_size=int(library_cfg.get("encode_batch_size", 1024)),
            max_queries=library_cfg.get("max_queries", 20000),
            temperature=_retrieval_temperature(config),
            popularity_coef=float(library_cfg.get("popularity_coef", 1.0)),
        ))

    health_cfg = config.get("representation_health") or {}
    if health_cfg.get("enabled", True):
        callbacks.append(RepresentationHealth(
            every_n_epochs=int(health_cfg.get("every_n_epochs", 5)),
            enable_after_epoch=int(health_cfg.get("enable_after_epoch", 0)),
            n_graphs=int(health_cfg.get("n_graphs", 4)),
            n_target_nodes=int(health_cfg.get("n_target_nodes", 2)),
            run_jacobian=bool(health_cfg.get("run_jacobian", True)),
        ))
    return callbacks


def get_trainer(config: DictConfig, ckpt_path: Path) -> pl.Trainer:
    kw        = OmegaConf.to_container(config.trainer_kwargs, resolve=True)
    logger    = get_logger(config,    ckpt_path)
    callbacks = get_callbacks(config, ckpt_path)
    return pl.Trainer(default_root_dir=str(ckpt_path),
                      logger=logger if logger is not None else True,
                      callbacks=callbacks,
                      **kw)


def load_weights_into(lightning_module, weights: Path, tag: str) -> int:
    """Load a checkpoint into whichever nested module actually owns the names.

    Resolve the target by KEY OVERLAP, never by assumed nesting.
    SaveBestModelCheckpoint saves ``pl_module.model.model`` (the nnet), so keys
    are ``nnet.*`` while the LightningModule's own are ``model.nnet.*``. Loading
    one into the other matches NOTHING, and with strict=False that is silent:
    the model stays randomly initialised and still produces a full set of
    plausible metrics. Not hypothetical -- it produced test/loss 14.29 against
    val 2.53 and r@1 0.0022 before being caught.
    """
    state = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    candidates = [("lightning_module", lightning_module)]
    inner = getattr(lightning_module, "model", None)
    if inner is not None:
        candidates.append(("lightning_module.model", inner))
        inner2 = getattr(inner, "model", None)
        if inner2 is not None:
            candidates.append(("lightning_module.model.model", inner2))
    keys = set(state)
    name, target, _ = max(
        ((n, m, len(keys & set(m.state_dict()))) for n, m in candidates),
        key=lambda c: c[2],
    )
    missing, unexpected = target.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    if loaded == 0:
        raise RuntimeError(
            f"[{tag}] {weights.name} shares NO parameter names with any of "
            f"{[n for n, _ in candidates]}; every key would be silently ignored "
            f"and the model would score as randomly initialised. "
            f"checkpoint[:3]={list(state)[:3]} "
            f"module[:3]={list(target.state_dict())[:3]}"
        )
    if missing or unexpected:
        logging.warning(
            f"[{tag}] partial state_dict match for {weights.name}: "
            f"{loaded}/{len(state)} tensors loaded, {len(missing)} missing, "
            f"{len(unexpected)} unexpected. "
            f"missing[:5]={list(missing)[:5]} unexpected[:5]={list(unexpected)[:5]}"
        )
    logging.info(f"[{tag}] loaded {loaded}/{len(state)} tensors from "
                 f"{weights.name} into {name} ({type(target).__name__})")
    return loaded


def train(config: DictConfig) -> None:
    logging.info(f"STARTED =====> Initializing Configuration  "
                 f"[seed={config.random_seed}]")
    config, base_path, ckpt_path = get_init_config(config)
    logging.info(f"FINISHED ====> Initializing Configuration")

    logging.info(f"STARTED =====> Initializing Dataset Module "
                 f"[{config.data_module}]")
    data_module = get_data_module(config, base_path)
    logging.info(f"FINISHED ====> Initializing Dataset Module")

    logging.info(f"STARTED =====> Initializing MolPLAtte Module "
                 f"[{config.nnet_module}]")
    nnet_module = get_nnet_module(config)
    logging.info(f"FINISHED ====> Initializing MolPLAtte Module")

    logging.info(f"STARTED =====> Wrapping Loss Module "
                 f"[{config.loss_module}]")
    loss_module = get_loss_module(config, nnet_module)
    logging.info(f"FINISHED ====> Wrapping Loss Module")

    logging.info(f"STARTED =====> Wrapping Lightning Module "
                 f"[{config.lightning_module}]")
    lightning_module = get_lightning_module(config, loss_module)
    logging.info(f"FINISHED ====> Wrapping Lightning Module")

    # Warm start. Distinct from Lightning's ckpt_path resume: this loads
    # WEIGHTS ONLY, leaving optimiser state, LR schedule and epoch counter
    # fresh, which is what finetuning onto a different corpus needs.
    init_from = config.get("init_weights_from")
    if init_from:
        init_from = Path(init_from)
        if not init_from.is_file():
            raise FileNotFoundError(f"init_weights_from: no checkpoint at {init_from}")
        logging.info(f"STARTED =====> Warm start from {init_from.name}")
        load_weights_into(lightning_module, init_from, "train")
        logging.info(f"FINISHED ====> Warm start")

    logging.info(f"STARTED =====> Fitting Trainer")
    trainer = get_trainer(config, ckpt_path)
    trainer.fit(lightning_module, datamodule=data_module)
    logging.info(f"FINISHED ====> Fitting Trainer")


def test(config: DictConfig) -> None:
    """Evaluate a trained checkpoint on the held-out test split.

    Mirrors ``train`` up to the Lightning wrapper, then loads weights and runs
    a single test pass. The split is the same deterministic 5% that ``train``
    never touches -- ``DataModuleConfig.seed`` drives the partition, so a test
    run must use the SAME seed as the run that produced the checkpoint or it
    will score on molecules that were trained on.

    ``checkpoint_path`` defaults to the file ``SaveBestModelCheckpoint`` writes
    for this ``experiment_name``. Those are plain ``state_dict`` files, not
    Lightning checkpoints, so they are loaded directly rather than through
    ``load_from_checkpoint``.
    """
    logging.info(f"STARTED =====> Initializing Configuration  "
                 f"[seed={config.random_seed}]")
    config, base_path, ckpt_path = get_init_config(config)
    logging.info(f"FINISHED ====> Initializing Configuration")

    data_module = get_data_module(config, base_path)
    nnet_module = get_nnet_module(config)
    loss_module = get_loss_module(config, nnet_module)
    lightning_module = get_lightning_module(config, loss_module)

    exp_name = config.get("experiment_name") or "molplatte"
    weights = config.get("checkpoint_path") or (ckpt_path / f"{exp_name}_best.pt")
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(
            f"no checkpoint at {weights}. Train first, or pass "
            f"checkpoint_path=/path/to/weights.pt"
        )
    load_weights_into(lightning_module, weights, "test")

    logging.info(f"STARTED =====> Testing on the held-out split")
    trainer = get_trainer(config, ckpt_path)
    results = trainer.test(lightning_module, datamodule=data_module)
    logging.info(f"FINISHED ====> Testing")
    for r in results or []:
        for k in sorted(r):
            logging.info(f"  {k:<44} {r[k]:.6f}")
    return results
