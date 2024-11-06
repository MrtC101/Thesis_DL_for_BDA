# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
"""
    All postprocessing pipeline file creation methods.
"""
import pandas as pd
import torch
from postprocessing.plots.plot_results import bbs_by_level_figures
from utils.common.pathManager import FilePath
from utils.loggers.console_logger import LoggerSingleton
from utils.loggers.table_print import to_table
from utils.metrics.metric_computer import MetricComputer
from utils.visualization.label_mask_visualizer import LabelMaskVisualizer
from utils.visualization.label_to_color import LabelDict


def save_df(metric_df: pd.DataFrame, pred_out: FilePath, file_name: str):
    """Save metrics in csv"""
    file = pred_out.join(file_name+".csv")
    metric_df.to_csv(path_or_buf=file, mode="w", index=False)
    file = pred_out.join(file_name+".tex")
    metric_df.to_latex(file)


def save_img(out_dir: FilePath, dis_id: str, tile_id: str, prefix: str, tile: torch.Tensor):
    """Saves a pre or post disaster image"""
    img = tile.numpy()
    path = out_dir.join(f"{dis_id}_{tile_id}_{prefix}_disaster.png")
    LabelMaskVisualizer.save_arr_img(img, path)


def save_mask(out_dir: FilePath, dis_id: str, tile_id: str, pred_mask: torch.Tensor):
    """Saves a predicted damage mask as a colored image"""
    pred_path = out_dir.join(f"{dis_id}_{tile_id}_pred_damage_mask.png")
    pred_img = LabelMaskVisualizer.draw_label_img(pred_mask)
    LabelMaskVisualizer.save_tensor_img(pred_img, pred_path)


def save_bbs(out_dir: FilePath, dis_id: str, tile_id: str, prefix: str, bbs_df: pd.DataFrame):
    """Saves the building's bounding box images for each class."""
    data = [{"Level": LabelDict().get_key_by_num(lab), "Count": 0} for lab in range(1, 5)]
    table = pd.DataFrame(data, columns=["Level", "Count"])
    for lab, n in bbs_df.value_counts(["label"]).items():
        table.loc[table["Level"] == lab[0], "Count"] = n

    folder = out_dir.join(f"{prefix}_bbs")
    folder.create_folder()
    save_df(table, folder, f"{prefix}_table.csv")
    bbs_by_level_figures(dis_id, tile_id, bbs_df, folder)
    return table


def save_metrics_and_matrices(out_dir: FilePath, px_conf_tot: pd.DataFrame,
                              px_multi_conf_tot: pd.DataFrame, obj_conf_tot: pd.DataFrame,
                              obj_multi_conf_tot: pd.DataFrame, dmg_labels: list):
    """
    Save the metrics and confusion matrices to the output directory.

    Args:
        out_dir (FilePath): Directory to save the results.
        px_conf_tot (pd.DataFrame): Total pixel confusion matrix.
        px_multi_conf_tot (pd.DataFrame): Total pixel multi confusion matrix.
        obj_conf_tot (pd.DataFrame): Total object confusion matrix.
        obj_multi_conf_tot (pd.DataFrame): Total object multi confusion matrix.
        dmg_labels (list): List of damage labels.
    """
    metric_dir = out_dir.join("metrics")
    metric_dir.create_folder()

    log = LoggerSingleton()

    log.info("Metrics for all predicted tiles".upper())
    px_metrics = MetricComputer.compute_eval_metrics(px_conf_tot, dmg_labels)
    log.info(to_table(curr_type="pixel", df=px_metrics, odd=True, decim_digits=5))

    log.info("Pixel Metrics")
    obj_metrics = MetricComputer.compute_eval_metrics(obj_conf_tot, dmg_labels)
    log.info(to_table(curr_type="object", df=obj_metrics, odd=True, decim_digits=5))

    save_df(px_metrics, metric_dir, "pixel_metrics")
    save_df(obj_metrics, metric_dir, "object_metrics")
    save_df(px_conf_tot, metric_dir, "pixel_confusion_matrix")
    save_df(px_multi_conf_tot, metric_dir, "pixel_multi_confusion_matrix")
    save_df(obj_conf_tot, metric_dir, "obj_confusion_matrix")
    save_df(obj_multi_conf_tot, metric_dir, "obj_multi_confusion_matrix")
