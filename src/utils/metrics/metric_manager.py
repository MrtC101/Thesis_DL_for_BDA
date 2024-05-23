import os
import sys

import pandas as pd

from utils.common.logger import LoggerSingleton
from utils.metrics.common import Level
from utils.metrics.metric_computer import MetricComputer
from utils.metrics.table_print import to_table
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
import enum
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
        self.px_dmg_metric = MetricComputer(Level.PX_BLD,self.dmg_labels, phase_context, static_context)
        self.px_bld_metric = MetricComputer(Level.PX_DMG,self.bld_labels, phase_context, static_context)
        self.obj_dmg_metric = MetricComputer(Level.OBJ_BLD,self.dmg_labels, phase_context, static_context)
        self.obj_bld_metric = MetricComputer(Level.OBJ_DMG,self.bld_labels, phase_context, static_context)

    def get_confusion_matrices_for(self, level: Level, args : dict):
        """Gets confusion matrices for a given level.

        Args:
            level (Level): The level for which to get confusion matrices (e.g., PX_BLD, PX_DMG, OBJ_BLD, OBJ_DMG).
            *args: Additional arguments required for computing confusion matrices.
        """
        if (level == Level.PX_BLD):
            return MatrixComputer.conf_mtrx_for_px_level(level, self.bld_labels, **args)
        elif (level == Level.PX_DMG):
            return MatrixComputer.conf_mtrx_for_px_level(level, self.dmg_labels, **args)
        elif (level == Level.OBJ_BLD):
            return MatrixComputer.conf_mtrx_for_obj_level(level, self.bld_labels, 3, **args)
        elif(level == Level.OBJ_DMG):
            return MatrixComputer.conf_mtrx_for_obj_level(level, self.dmg_labels, 3, **args)
        else:
            raise Exception(f"{level} Not Implemented")

    def compute_metrics_for(self, level: Level, conf_df):
        """Computes metrics for a given level.

        Args:
            level (Level): The level for which to compute metrics (e.g., PX_BLD, PX_DMG, OBJ_BLD, OBJ_DMG).
            *args: Additional arguments required for computing metrics.
        """
        if (level == Level.PX_BLD):
            return self.px_bld_metric.compute_metrics(conf_df)
        elif (level == Level.PX_DMG):
            return self.px_dmg_metric.compute_metrics(conf_df)
        elif (level == Level.OBJ_BLD):
            return  self.obj_bld_metric.compute_metrics(conf_df)
        elif (level == Level.OBJ_DMG):
            return  self.obj_dmg_metric.compute_metrics(conf_df)
        else:
            raise Exception(f"{level} Not Implemented")
        
    
    def log_metrics(self, phase, tb_log, metrics: dict[pd.DataFrame]):
        """Logs evaluation metrics using the provided logger."""
        metric_df : pd.DataFrame
        log = LoggerSingleton()
        log.info(f"--{phase.upper()} METRICS--") 
        for key, metric_df in metrics.items():
            log.info(to_table(curr_type=key,df=metric_df,odd=True,decim_digits=5))
            for index, row in metric_df.iterrows():
                msg = f"{phase}/{key}_metrics"
                tb_log.add_scalars(msg, dict(row), int(row["epoch"]))