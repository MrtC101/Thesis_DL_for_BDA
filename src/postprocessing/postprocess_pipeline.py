import math
import os
import sys
from os.path import join
import pandas as pd
import numpy as np
import pandas as pd
from shapely.wkt import loads

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.metrics.matrix_computer import MatrixComputer
from utils.metrics.metric_computer import MetricComputer
from utils.metrics.metric_manager import MetricManager
from postprocessing.ploting.bounding_boxes import get_bbs_form_json, get_bbs_form_mask
from utils.loggers.console_logger import LoggerSingleton
from utils.datasets.predicted_dataset import PredictedDataset
from utils.visualization.plot_results import generate_figures


def save_metric(px_metric: pd.DataFrame, pred_out, file_prefix="pixel"):
    """Save metrics in csv"""
    file = os.path.join(pred_out, f'{file_prefix}_metrics.csv')
    px_metric.to_csv(path_or_buf=file, mode="w", index=False)


def postprocess(split_raw_json_path, definitive_folder, label_map_json, save_path):
    """Implements the postprocessing pipeline"""
    log = LoggerSingleton()
    log.name = "Postprocessing"
    dmg_labels = [1, 2, 3, 4]
    predicted_patches_folder_path = join(definitive_folder, "test_pred_masks")
    predicted_dataset = PredictedDataset(
        split_raw_json_path, predicted_patches_folder_path)
    for dis_id, tile_id, tile_dict, pred_mask in predicted_dataset:
        pred_out = os.path.join(save_path, f"{dis_id}_{tile_id}")
        os.makedirs(pred_out, exist_ok=True)
        px_conf_mtrx = MatrixComputer.tile_px_conf_mtrx(tile_dict["bld_mask"],
                                                        tile_dict["dmg_mask"],
                                                        pred_mask, dmg_labels)
        px_metric = MetricComputer(dmg_labels).\
            compute_eval_metrics(px_conf_mtrx)
        save_metric(px_metric, pred_out, file_prefix="pixel")

        obj_conf_mtrx = MatrixComputer.tile_obj_conf_mtrx(tile_dict["bld_mask"],
                                                          pred_mask, dmg_labels)
        obj_metric = MetricComputer(dmg_labels).\
            compute_eval_metrics(obj_conf_mtrx)
        save_metric(obj_metric, pred_out, file_prefix="object")

        gt_bbs_df = get_bbs_form_json(tile_dict["dmg_json"])
        gt_values = list(gt_bbs_df.value_counts(subset=["label"]).items())
        gt_table = pd.DataFrame([(val[0][0], val[1])
                                for val in gt_values], columns=["Level", "Count"])

        pd_bbs_df = get_bbs_form_mask(pred_mask, dmg_labels)
        pd_values = list(pd_bbs_df.value_counts(subset=["label"]).items())
        pd_table = pd.DataFrame([(val[0][0], val[1])
                                for val in pd_values], columns=["Level", "Count"])

        generate_figures(dis_id, tile_id, tile_dict, pred_mask,
                         label_map_json, (gt_bbs_df, gt_table),
                         (pd_bbs_df, pd_table), pred_out)
