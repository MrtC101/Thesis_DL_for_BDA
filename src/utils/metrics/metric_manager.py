# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys

import pandas as pd
import torch
from utils.common.logger import LoggerSingleton
from utils.metrics.common import Level
from utils.metrics.metric_computer import MetricComputer
from utils.metrics.table_print import to_table
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
import concurrent.futures
from utils.metrics.matrix_computer import MatrixComputer


class MetricManager:
    """
     A class to manage metrics for building and damage classification.

    This class initializes metric computers for pixel and object-level metrics
    for both building and damage labels. It provides methods to get confusion
    matrices and compute metrics for different levels (pixel or object) and
    types (building or damage).
    """

    def __init__(self, phase_context, static_context) -> None:
        self.bld_labels = static_context['labels_set_bld']
        self.dmg_labels = static_context['labels_set_dmg']
        self.px_dmg_metric = \
            MetricComputer(Level.PX_BLD, self.dmg_labels,
                           phase_context, static_context)
        self.px_bld_metric = \
            MetricComputer(Level.PX_DMG, self.bld_labels,
                           phase_context, static_context)
        self.obj_dmg_metric = \
            MetricComputer(Level.OBJ_BLD, self.dmg_labels,
                           phase_context, static_context)
        self.obj_bld_metric = \
            MetricComputer(Level.OBJ_DMG, self.bld_labels,
                           phase_context, static_context)

    def get_confusion_matrices_for(self, level: Level, args: dict):
        """Gets confusion matrices for a given level.

        Args:
            level (Level): The level for which to get confusion matrices
            (e.g., PX_BLD, PX_DMG, OBJ_BLD, OBJ_DMG).
            *args: Additional arguments required for computing confusion matrices.
        """
        if (level == Level.PX_BLD):
            return MatrixComputer.conf_mtrx_for_px_level(level, self.bld_labels, **args)
        elif (level == Level.PX_DMG):
            return MatrixComputer.conf_mtrx_for_px_level(level, self.dmg_labels, **args)
        elif (level == Level.OBJ_BLD):
            return MatrixComputer.conf_mtrx_for_obj_level(level, self.bld_labels, 3, **args)
        elif (level == Level.OBJ_DMG):
            return MatrixComputer.conf_mtrx_for_obj_level(level, self.dmg_labels, 3, **args)
        else:
            raise Exception(f"{level} Not Implemented")

    def compute_metrics_for(self, level: Level, conf_df):
        """Computes metrics for a given level.

        Args:
            level (Level): The level for which to compute metrics
            (e.g., PX_BLD, PX_DMG, OBJ_BLD, OBJ_DMG).
            *args: Additional arguments required for computing metrics.
        """
        if (level == Level.PX_BLD):
            return self.px_bld_metric.compute_metrics(conf_df)
        elif (level == Level.PX_DMG):
            return self.px_dmg_metric.compute_metrics(conf_df)
        elif (level == Level.OBJ_BLD):
            return self.obj_bld_metric.compute_metrics(conf_df)
        elif (level == Level.OBJ_DMG):
            return self.obj_dmg_metric.compute_metrics(conf_df)
        else:
            raise Exception(f"{level} Not Implemented")

    def log_metrics(self, phase, tb_log, metrics: dict[pd.DataFrame]):
        """Logs evaluation metrics using the provided logger."""
        metric_df: pd.DataFrame
        log = LoggerSingleton()
        log.info(f"--{phase.upper()} METRICS--")
        for key, metric_df in metrics.items():
            log.info(to_table(curr_type=key, df=metric_df,
                     odd=True, decim_digits=5))
            for index, row in metric_df.iterrows():
                msg = f"{phase}/{key}_metrics"
                tb_log.add_scalars(msg, dict(row), int(row["epoch"]))

    def compute_epoch_metrics(self, phase, tb_log, epoch, confusion_matrices_df: pd.DataFrame,
                              levels = [Level.PX_DMG, Level.PX_BLD, Level.OBJ_DMG, Level.OBJ_BLD],
                              paralelism=False):
        """Computes metrics for damage and building classification
        for the current phase in the current epoch.

        Args:
            confusion_matrices (dict): Dictionary containing confusion matrices.
            epoch (int): The current epoch number.

        Returns:
            dict: Dictionary containing computed metrics for damage and building classification.
        """
  
        def compute_metrics_for_level(lvl, mtrx):
            matrix = self.compute_metrics_for(lvl, confusion_matrices_df[mtrx])
            matrix.insert(0, "epoch", epoch)
            return matrix

        metrics = {}
        if paralelism:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_key_lvl = {
                    executor.submit(compute_metrics_for_level, lvl,  lvl.value["matrix_key"]):
                    (lvl) for lvl in levels
                }
                for future in concurrent.futures.as_completed(future_to_key_lvl):
                    lvl = future_to_key_lvl[future]
                    metrics[lvl.value["metric_key"]] = future.result()
        else:
            for lvl in levels:
                metrics[lvl.value["metric_key"]] = \
                    compute_metrics_for_level(lvl, lvl.value["matrix_key"])
        self.log_metrics(phase=phase, tb_log=tb_log, metrics=metrics)
        return metrics

    def compute_confusion_matrices(self, y_seg: torch.Tensor, y_cls: torch.Tensor,
                                   pred_y_seg: torch.Tensor, pred_y_cls: torch.Tensor,
                                   batch_idx: int,levels : list[Level] = \
                                   [Level.PX_DMG, Level.PX_BLD, Level.OBJ_DMG, Level.OBJ_BLD],
                                   paralelism=False, *kwargs) -> dict:
        """
        Computes confusion matrices for damage and building classification at different levels.

        Args:
            y_seg (torch.Tensor): Ground truth segmentation tensor.
            y_cls (torch.Tensor): Ground truth classification tensor.
            pred_y_seg (torch.Tensor): Predicted segmentation tensor.
            pred_y_cls (torch.Tensor): Predicted classification tensor.
            batch_idx (int): Index of the current batch.
            *kwargs: Additional arguments.

        Returns:
            dict: Dictionary containing confusion matrices and batch identifier.
        """

        def get_confusion_matrix_for_level(lvl, data, batch_idx):
            matrix = self.get_confusion_matrices_for(lvl, data)
            matrix.insert(0, "batch_id", batch_idx)
            return matrix

        data = {"pred_bld_mask": pred_y_seg,
                "pred_dmg_mask": pred_y_cls,
                "y_bld_mask": y_seg,
                "y_dmg_mask": y_cls}
        matrices = {}
        if paralelism:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_key_lvl = {
                    executor.submit(get_confusion_matrix_for_level, lvl, data, batch_idx):
                    (lvl) for lvl in levels
                }
                for future in concurrent.futures.as_completed(future_to_key_lvl):
                    lvl = future_to_key_lvl[future]
                    matrices[lvl.value["matrix_key"]] = future.result()
        else:
            for lvl in levels:
                matrices[lvl.value["matrix_key"]] = get_confusion_matrix_for_level(lvl, data, batch_idx)
        return matrices