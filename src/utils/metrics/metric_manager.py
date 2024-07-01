# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys

import pandas as pd
import torch
from utils.loggers.console_logger import LoggerSingleton
from utils.metrics.common import Level
from utils.metrics.metric_computer import MetricComputer
from utils.loggers.table_print import to_table
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
import concurrent.futures
from utils.metrics.matrix_computer import MatrixComputer

parallelism = bool(os.environ["parallelism"])

class MetricManager:
    """
     A class to manage metrics for building and damage classification.

    This class initializes metric computers for pixel and object-level metrics
    for both building and damage labels. It provides methods to get confusion
    matrices and compute metrics for different levels (pixel or object) and
    types (building or damage).
    """

    def __init__(self, bld_labels, dmg_labels) -> None:
        self.bld_labels = bld_labels
        self.dmg_labels = dmg_labels
        self.px_dmg_metric = MetricComputer(self.dmg_labels)
        self.px_bld_metric = MetricComputer(self.bld_labels)
        self.obj_dmg_metric = MetricComputer(self.dmg_labels)
        self.obj_bld_metric = MetricComputer(self.bld_labels)


    def parallelism_by_level(self, levels, lvl_key, func, *params):
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            for lvl in levels:
                key = lvl.value[lvl_key]
                future = executor.submit(func, lvl, *params)
                futures[future] = key 
            for future in concurrent.futures.as_completed(futures):
                key = futures[future] 
                results[key] = future.result()
        return results

    def get_confusion_matrices_for(self, level: Level, args: dict):
        """Gets confusion matrices for a given level.

        Args:
            level (Level): The level for which to get confusion matrices
            (e.g., PX_BLD, PX_DMG, OBJ_BLD, OBJ_DMG).
            *args: Additional arguments required for computing confusion matrices.
        """
        if (level == Level.PX_BLD):
            return MatrixComputer.patches_px_conf_mtrx(level, self.bld_labels, **args)
        elif (level == Level.PX_DMG):
            return MatrixComputer.patches_px_conf_mtrx(level, self.dmg_labels, **args)
        elif (level == Level.OBJ_BLD):
            return MatrixComputer.patches_obj_conf_mtrx(level, self.bld_labels, 3, **args)
        elif (level == Level.OBJ_DMG):
            return MatrixComputer.patches_obj_conf_mtrx(level, self.dmg_labels, 3, **args)
        
    def compute_confusion_matrices(self ,batch_idx,
                                gt_bld_mask : torch.Tensor, gt_dmg_mask : torch.Tensor,
                                pd_bld_mask : torch.Tensor, pd_dmg_mask : torch.Tensor,
                                levels=[Level.PX_DMG, Level.PX_BLD, Level.OBJ_DMG, Level.OBJ_BLD]
                                ) -> dict:
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

        def get_confusion_matrix_for_level(lvl, batch_idx):
            matrix = self.get_confusion_matrices_for(lvl, { "gt_bld_mask": gt_bld_mask,
                                                            "gt_dmg_mask": gt_dmg_mask,
                                                            "pd_bld_mask": pd_bld_mask,
                                                            "pd_dmg_mask": pd_dmg_mask })
            matrix.insert(0, "batch_id", batch_idx)
            return matrix

        matrices = {}
        if parallelism:
            matrices = self.\
                parallelism_by_level(levels, "matrix_key", get_confusion_matrix_for_level,
                                      batch_idx)
        else:
            for lvl in levels:
                key = lvl.value["matrix_key"]
                matrices[key] = get_confusion_matrix_for_level(lvl, batch_idx)
        return matrices

   

    def compute_metrics_for(self, level : Level, conf_df : pd.Series):
        """Computes metrics for a given level."""
        if (level == Level.PX_BLD):
            return self.px_bld_metric.compute_metrics(conf_df)
        elif (level == Level.PX_DMG):
            return self.px_dmg_metric.compute_metrics(conf_df)
        elif (level == Level.OBJ_BLD):
            return self.obj_bld_metric.compute_metrics(conf_df)
        elif (level == Level.OBJ_DMG):
            return self.obj_dmg_metric.compute_metrics(conf_df)

    def compute_epoch_metrics(self ,epoch : int, confusion_matrices: list,
                            levels=[Level.PX_DMG, Level.PX_BLD, Level.OBJ_DMG, Level.OBJ_BLD]
                            ) -> dict:
        """Computes metrics for damage and building classification
        for the current phase in the current epoch.

        Args:
            confusion_matrices (dict): Dictionary containing confusion matrices.
            epoch (int): The current epoch number.

        Returns:
            dict: Dictionary containing computed metrics for damage and building classification.
        """
        confusion_matrices_df = pd.DataFrame(confusion_matrices)

        def compute_metrics_for_level(lvl):
            key = lvl.value["matrix_key"]
            matrix = self.compute_metrics_for(lvl, confusion_matrices_df[key])
            matrix.insert(0, "epoch", epoch)
            return matrix

        metrics = {}
        if parallelism:
            metrics = self.\
                parallelism_by_level(levels, "metric_key", compute_metrics_for_level)
        else:
            for lvl in levels:
                key = lvl.value["metric_key"] 
                metrics[key] = compute_metrics_for_level(lvl)
        return metrics
    
    @staticmethod
    def log_metrics(phase, tb_log, metrics: dict[pd.DataFrame]):
        """Logs evaluation metrics using the provided logger."""
        metric_df: pd.DataFrame
        log = LoggerSingleton()
        log.info(f"--{phase.upper()} METRICS--")
        for key, metric_df in metrics.items():
            log.info(to_table(curr_type=key, df=metric_df,odd=True, decim_digits=5))
            for index, row in metric_df.iterrows():
                msg = f"{phase}/{key}_metrics"
                tb_log.add_scalars(msg, dict(row), int(row["epoch"]))

    @staticmethod
    def save_metrics(metrics, metric_dir,file_prefix):
        """Save metrics in csv"""
        # save evalution metrics
        for epoch in range(len(metrics)):
            for key, met in metrics[epoch].items():
                mode = "w" if not epoch > 0 else "a"
                header = not epoch > 0
                met.to_csv(os.path.join(
                    metric_dir, f'{file_prefix}_{key}.csv'),
                      mode=mode, header=header, index=False)

    @staticmethod
    def save_loss(loss_metrics,metric_dir,filename):
        """Save metrics in csv"""
        path = os.path.join(metric_dir,"loss.csv")
        df = pd.DataFrame(loss_metrics)
        df.to_csv(path, mode="w", index=False)    