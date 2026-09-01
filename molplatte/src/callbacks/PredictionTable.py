from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class PredictionTable(pl.Callback):
    """Dump the top-K R-group retrieval neighbours for each validation query.

    One row per **detached R-group** (i.e. per linker), matching
    :class:`FAISSRetrieval` and MolPLA's per-linker retrieval head.

    Batch contract
    --------------
    required
        ``query_projection``  (n_R, D)   per-linker query embedding
        ``rgroup_projection`` (n_R, D)   pooled R-group embedding, same order
        ``R_hashes``          list[str]  length n_R, the retrieval target key
    optional (used for readable row identity; blank if absent)
        ``rgroup_to_sample``  LongTensor (n_R,) row -> instance index
        ``instance_ids``      list[str]  length = #instances
        ``P_hashes``          list[str]  length = #instances, hash of ``P``

    Writes a CSV **only on epochs where the monitored validation metric
    improves** (same monitor/mode convention as
    :class:`SaveBestModelCheckpoint`), so a long run leaves a handful of
    tables rather than one per epoch. Files land in
    ``{out_dir}/prediction_tables/epoch_{NNN}_{monitor}_{value}.csv``.

    If a wandb logger happens to be attached the same rows are also
    pushed as a ``wandb.Table``; that path is strictly optional and its
    failure never costs you the CSV.
    """

    def __init__(self,
                 k:                  int = 5,
                 max_rows:           int = 500,
                 log_every_n_epochs: int = 1,
                 enable_after_epoch: int = 0,
                 out_dir:            Optional[str] = None,
                 monitor:            str = "val/loss",
                 mode:               str = "min"):
        super().__init__()
        assert mode in ("min", "max")
        self.k                  = k
        self.max_rows           = max_rows
        self.log_every_n_epochs = log_every_n_epochs
        self.enable_after_epoch = int(enable_after_epoch)
        self.out_dir            = out_dir
        self.monitor            = monitor
        self.mode               = mode
        self.best               = float("inf") if mode == "min" else float("-inf")
        self._reset()

    def _is_better(self, current: float) -> bool:
        return (current < self.best) if self.mode == "min" else (current > self.best)

    def _table_dir(self, trainer) -> Path:
        base = self.out_dir
        if base is None:
            base = getattr(trainer, "default_root_dir", None) or os.getcwd()
        return Path(base) / "prediction_tables"

    def _reset(self):
        self._rows: List[Dict] = []
        self._q:    List[torch.Tensor] = []
        self._g:    List[torch.Tensor] = []

    def _gated(self, trainer) -> bool:
        # A test pass has no epoch schedule -- current_epoch is 0 for a freshly
        # constructed trainer loading a bare state_dict, so enable_after_epoch
        # would gate away the one table the run exists to produce.
        if trainer.testing:
            return False
        return trainer.current_epoch < self.enable_after_epoch

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self._gated(trainer):
            return
        q = batch.get("query_projection")
        g = batch.get("rgroup_projection")
        hashes = batch.get("R_hashes")
        if q is None or g is None or hashes is None:
            return

        # Row identity. MolDAM keyed rows by ZINC id + core hash; MolPLAtte's
        # unit is a *linker*, and there are J of those against B instances, so
        # the row must be mapped back through joint_sample.
        #
        # This previously read "rgroup_to_sample", which this collate does not
        # emit -- the key is "joint_sample". The lookup silently fell back to
        # s = i, labelling linker row i with instance_ids[i]. Those are different
        # axes (J != B), so query_instance_id was MISLABELLED rather than blank,
        # which is the worse failure: a wrong id reads as a real one.
        #
        # Fail loudly on a missing/mismatched map instead of degrading, for the
        # same reason.
        j2s      = batch.get("joint_sample")
        inst_ids = batch.get("instance_ids")
        if j2s is None or inst_ids is None or len(j2s) != len(hashes):
            log = logging.getLogger(__name__)
            log.warning(
                "[PredictionTable] cannot map linkers to instances "
                "(joint_sample=%s, hashes=%d) -- skipping this batch rather than "
                "emitting unlabelled rows",
                None if j2s is None else len(j2s), len(hashes),
            )
            return
        for i, h in enumerate(hashes):
            s = int(j2s[i])
            self._rows.append({
                "instance_id": inst_ids[s] if s < len(inst_ids) else "",
                "target_hash": str(h),
            })
        self._q.append(q.detach().cpu())
        self._g.append(g.detach().cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        self.on_validation_epoch_end(trainer, pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                          dataloader_idx=0):
        self.on_validation_batch_end(trainer, pl_module, outputs, batch,
                                     batch_idx, dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        log = logging.getLogger(__name__)
        if trainer.sanity_checking or self._gated(trainer):
            self._reset(); return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            self._reset(); return
        if trainer.global_rank != 0:
            self._reset(); return
        if not self._rows:
            return

        # Only materialise a table when the watched metric improves — keeps a
        # 120-epoch run down to a handful of CSVs instead of one per epoch.
        # On a test pass there is no epoch sequence and nothing to improve on,
        # so the monitor gate below would drop the only table we want.
        if trainer.testing:
            current = None
        else:
            current = trainer.callback_metrics.get(self.monitor)
        if current is None and not trainer.testing:
            log.info(f"[PredictionTable] monitor {self.monitor!r} "
                     f"not in callback_metrics — skipping table this epoch")
            self._reset(); return
        if not trainer.testing:
            current = float(current)
            if not self._is_better(current):
                self._reset(); return
            self.best = current

        import faiss
        q = torch.cat(self._q).float().numpy().astype("float32")
        g = torch.cat(self._g).float().numpy().astype("float32")
        valid = (np.linalg.norm(q, axis=1) > 1e-6) & (np.linalg.norm(g, axis=1) > 1e-6)
        q = q[valid]; g = g[valid]
        rows_kept = [self._rows[i] for i, v in enumerate(valid) if v]
        N, D   = q.shape
        if N < 2:
            self._reset(); return

        # Retrieve over DISTINCT target chemistries, not entries. On the entry
        # gallery a common R-group holds thousands of rows, so once it is
        # ranked first its duplicates fill every remaining slot: measured on
        # MolDAM v29a/GINEConv, 56.8% of queries had a top-5 that was one
        # chemistry repeated five times (median 1 distinct hash across 5
        # slots). Those rows are unreadable. One representative per
        # target_hash instead.
        seen: Dict[str, int] = {}
        rep_of_row = []
        for r in rows_kept:
            th = r["target_hash"]
            if th not in seen:
                seen[th] = len(seen)
            rep_of_row.append(seen[th])
        rep_of_row = np.asarray(rep_of_row)
        first_row = np.zeros(len(seen), dtype=np.int64)
        got = np.zeros(len(seen), dtype=bool)
        for i, c in enumerate(rep_of_row):
            if not got[c]:
                first_row[c] = i
                got[c] = True

        K = min(self.k, len(seen))
        idx_g = faiss.IndexFlatIP(D); idx_g.add(np.ascontiguousarray(g[first_row]))
        sim_qg, top_chem = idx_g.search(q, K)
        # map chemistry rank back to a representative row for display
        top_qg = first_row[top_chem]

        n_rows = min(self.max_rows, len(rows_kept), N)
        cols = ["epoch", self.monitor,
                "query_instance_id", "target_hash",
                "retrieved_instance_ids", "retrieved_target_hashes",
                "retrieved_sims", "positive_rank"]
        data = []
        for i in range(n_rows):
            nbr = top_qg[i, :K].tolist()
            sims = sim_qg[i, :K].tolist()
            # Rank of the query's own CHEMISTRY among the retrieved distinct
            # targets. Index matching would be wrong here: the correct answer
            # is the target_hash, and thousands of rows may carry it.
            chem = rep_of_row[i]
            ranked = top_chem[i, :K].tolist()
            pos_rank = ranked.index(chem) + 1 if chem in ranked else ""
            data.append([
                trainer.current_epoch,
                # None on a test pass: there is no monitored validation metric
                # to record. Writing "" keeps the column present and the CSV
                # shape stable rather than crashing on a NoneType format.
                f"{current:.6f}" if current is not None else "",
                rows_kept[i]["instance_id"],
                rows_kept[i]["target_hash"],
                # " | " between retrieved items: kept from MolDAM even though
                # MolPLAtte's target_hash is a single hash rather than a
                # comma-joined set, so the column stays splittable if a future
                # variant goes back to sets.
                " | ".join(rows_kept[j]["instance_id"] for j in nbr),
                " | ".join(rows_kept[j]["target_hash"] for j in nbr),
                " | ".join(f"{s:.4f}" for s in sims),
                pos_rank,
            ])

        out_dir = self._table_dir(trainer)
        out_dir.mkdir(parents=True, exist_ok=True)
        metric_tag = self.monitor.replace("/", "_").replace("@", "at")
        if trainer.testing:
            # No monitored value on a test pass, and current_epoch is whatever
            # the checkpoint left behind -- name it for what it is instead.
            out_path = out_dir / "test.csv"
        else:
            out_path = (out_dir /
                        f"epoch_{trainer.current_epoch:03d}_{metric_tag}_{current:.4f}.csv")
        try:
            with open(out_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(cols)
                w.writerows(data)
            log.info(f"[PredictionTable] wrote {n_rows} rows "
                     f"(corpus N={N}, {self.monitor}={current:.4f}) → {out_path}")
        except Exception as e:
            log.warning(f"[PredictionTable] CSV write failed: {e}")

        # Optional wandb mirror — never allowed to cost us the CSV above.
        try:
            import wandb
            logger = trainer.logger
            if logger is not None and hasattr(logger, "experiment"):
                logger.experiment.log({
                    "val/prediction_table": wandb.Table(columns=cols, data=data),
                    "trainer/global_step":  trainer.global_step,
                })
        except Exception:
            pass
        self._reset()
