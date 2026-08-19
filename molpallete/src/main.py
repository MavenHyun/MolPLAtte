from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

import data_modules as dm
from data_modules import MolPalleteDataModule       # noqa: F401  (resolved via getattr)

import nnet_modules      as nm
import loss_modules      as lossm
import lightning_modules as lm
from nnet_modules      import MolPallete                   # noqa: F401  (resolved via getattr)
from loss_modules      import LossModuleMolPallete         # noqa: F401  (resolved via getattr)
from lightning_modules import MolPalleteLightningModule    # noqa: F401  (resolved via getattr)

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

    base_path = Path(config.master_path)
    ckpt_path = base_path / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    return config, base_path, ckpt_path

def get_data_module(config: DictConfig, base_path: Path) -> MolPalleteDataModule:
    if config.data_module_kwargs.get("dataset_path") is None:
        config.data_module_kwargs.dataset_path = str(base_path / "preprocessed" / "molpallete")

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
    logger = pl.loggers.WandbLogger(
        project=wcfg.get("project"),
        name=wcfg.get("name"),
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
    on ``decomposition_paradigm``; MolPallete has a single view scheme (MolPLA's
    G/P/R/Q views), so the branch is gone and the flag is read directly.
    """
    return bool(_retrieval_loss_kwargs(config).get("logq_correction", True))


def _retrieval_temperature(config: DictConfig) -> float:
    return float(_retrieval_loss_kwargs(config).get("temperature", 0.01))


def get_callbacks(config: DictConfig, ckpt_path: Path) -> List[pl.Callback]:
    exp_name = config.get("experiment_name")
    ckpt_filename = f"{exp_name}_best.pt" if exp_name else "molpallete_best.pt"
    es_cfg = config.get("early_stopping", {}) or {}
    es_patience = int(es_cfg.get("patience", 10))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    es_monitor  = str(es_cfg.get("monitor", "val/loss"))
    es_mode     = str(es_cfg.get("mode", "min"))
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
        pl.callbacks.EarlyStopping(monitor=es_monitor, mode=es_mode,
                                   patience=es_patience,
                                   min_delta=es_min_delta,
                                   check_finite=True,
                                   verbose=True),
        # Training and evaluation must agree on what the similarity estimates.
        # With logq_correction the critic learns log p(k|q) directly, so FAISS
        # ranks raw similarity. Without it the critic learns PMI
        # (log p(k|q) - log p(k)) and the popularity term has to be added back
        # at scoring, or r@k underreports badly -- measured on MolDAM
        # v29a/GINEConv, 0.3805 vs 0.6646. That number is MolDAM's ZINC
        # anchored corpus, not MolPallete's; the *mechanism* carries over
        # (skewed R-group vocabulary + in-batch InfoNCE), the magnitude does
        # not. Deriving one flag from the other prevents the mismatch either way.
        FAISSRetrieval(enable_after_epoch=3,
                       popularity_correction=not _logq_enabled(config),
                       temperature=_retrieval_temperature(config)),
        # MolDAM's MolecularReassembly callback is intentionally absent:
        # MolPallete has no assembly objective (MolPLA's three losses are all
        # contrastive), so there is nothing for it to score.
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


def train(config: DictConfig) -> None:
    logging.info(f"STARTED =====> Initializing Configuration  "
                 f"[seed={config.random_seed}]")
    config, base_path, ckpt_path = get_init_config(config)
    logging.info(f"FINISHED ====> Initializing Configuration")

    logging.info(f"STARTED =====> Initializing Dataset Module "
                 f"[{config.data_module}]")
    data_module = get_data_module(config, base_path)
    logging.info(f"FINISHED ====> Initializing Dataset Module")

    logging.info(f"STARTED =====> Initializing MolPallete Module "
                 f"[{config.nnet_module}]")
    nnet_module = get_nnet_module(config)
    logging.info(f"FINISHED ====> Initializing MolPallete Module")

    logging.info(f"STARTED =====> Wrapping Loss Module "
                 f"[{config.loss_module}]")
    loss_module = get_loss_module(config, nnet_module)
    logging.info(f"FINISHED ====> Wrapping Loss Module")

    logging.info(f"STARTED =====> Wrapping Lightning Module "
                 f"[{config.lightning_module}]")
    lightning_module = get_lightning_module(config, loss_module)
    logging.info(f"FINISHED ====> Wrapping Lightning Module")

    logging.info(f"STARTED =====> Fitting Trainer")
    trainer = get_trainer(config, ckpt_path)
    trainer.fit(lightning_module, datamodule=data_module)
    logging.info(f"FINISHED ====> Fitting Trainer")


def test(config: DictConfig) -> None:
    """Placeholder -- wire in once a checkpointed model exists."""
    logging.info("[stub] test() not yet implemented")
    return
