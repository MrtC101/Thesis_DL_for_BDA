import os
import sys
from os.path import join
import pandas as pd

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.visualization.label_mask_visualizer import LabelMaskVisualizer
from utils.visualization.label_to_color import LabelDict
from utils.metrics.matrix_computer import MatrixComputer
from utils.metrics.metric_manager import MetricComputer
from utils.loggers.console_logger import LoggerSingleton
from utils.datasets.predicted_dataset import PredictedDataset
from postprocessing.plots.plot_results import bbs_by_level_figures, \
    comparative_figure, superposed_img
from postprocessing.bbs.bounding_boxes import get_bbs_from_json, \
    get_bbs_from_mask


def save_df(px_metric: pd.DataFrame, pred_out, file_name):
    """Save metrics in csv"""
    file = os.path.join(pred_out, file_name+".csv")
    px_metric.to_csv(path_or_buf=file, mode="w", index=False)
    file = os.path.join(pred_out, file_name+".tex")
    px_metric.to_latex(file, index=False)


def save_img(save_path, dis_id, tile_id, prefix, tile):
    pre_img = tile.numpy()
    pre_path = os.path.join(
        save_path, f"{dis_id}_{tile_id}_{prefix}_disaster.png")
    LabelMaskVisualizer.save_arr_img(pre_img, pre_path)


def save_mask(save_path, dis_id, tile_id, pred_mask):
    pred_path = os.path.join(
        save_path, f"{dis_id}_{tile_id}_pred_damage_mask.png")
    pred_img = LabelMaskVisualizer().draw_label_img(pred_mask)
    LabelMaskVisualizer.save_arr_img(pred_img, pred_path)


def save_bbs(dis_id, tile_id, bbs_df, prefix, save_path):
    values = list(bbs_df.value_counts(subset=["label"]).items())
    table = pd.DataFrame([(val[0][0], val[1]) for val in values],
                         columns=["Level", "Count"])
    folder = os.path.join(save_path, f"{prefix}_bbs")
    os.makedirs(folder, exist_ok=True)
    save_df(table, folder, f"{prefix}_table.csv")
    bbs_by_level_figures(dis_id, tile_id, bbs_df, folder)
    return table


def pixel_analysis(tile_dict, pred_mask, dmg_labels, pred_out):
    conf_mtrx = MatrixComputer.tile_px_conf_mtrx(tile_dict["bld_mask"],
                                                 tile_dict["dmg_mask"],
                                                 pred_mask, dmg_labels)
    save_df(conf_mtrx, pred_out, "pixel_confusion_matrix")

    multi_conf_mtrx = MatrixComputer.px_multiclass_conf_mtrx(
        tile_dict["dmg_mask"], pred_mask, dmg_labels)
    save_df(multi_conf_mtrx, pred_out, "pixel_multiclass_confusion_matrix")

    metrics = MetricComputer.compute_eval_metrics(conf_mtrx, dmg_labels)
    save_df(metrics, pred_out, "pixel_metrics")

    return conf_mtrx, multi_conf_mtrx


def object_analysis(tile_dict, pred_mask, dmg_labels, pred_out):
    conf_mtrx, multi_conf_mtrx = MatrixComputer.\
        tile_obj_conf_matrices(tile_dict["dmg_mask"], pred_mask, dmg_labels)
    save_df(conf_mtrx, pred_out, "object_confusion_matrix")
    save_df(multi_conf_mtrx, pred_out, "object_multiclass_confusion_matrix")

    metrics = MetricComputer.compute_eval_metrics(conf_mtrx, dmg_labels)
    save_df(metrics, pred_out, "object_metrics")

    return conf_mtrx, multi_conf_mtrx


def make_figures(save_path, dis_id, tile_id, tile_dict, pred_mask):
    save_img(save_path, dis_id, tile_id, "pre", tile_dict["pre_img"])
    save_img(save_path, dis_id, tile_id, "post", tile_dict["post_img"])
    save_mask(save_path, dis_id, tile_id, pred_mask)
    if tile_dict["dmg_json"] is None:
        gt_bbs_df = get_bbs_from_json(tile_dict["dmg_json"])
    else:
        gt_bbs_df = get_bbs_from_json(tile_dict["dmg_mask"])
    gt_table = save_bbs(dis_id, tile_id, gt_bbs_df, "gt", save_path)
    pd_bbs_df = get_bbs_from_mask(pred_mask)
    pd_table = save_bbs(dis_id, tile_id, pd_bbs_df, "pd", save_path)
    comparative_figure(dis_id, tile_id,  tile_dict["pre_img"],
                       tile_dict["post_img"], pred_mask,
                       gt_table, pd_table, save_path)
    superposed_img(dis_id, tile_id,
                   tile_dict["pre_img"], tile_dict["post_img"], save_path)


def add_confusion_matrices(conf_mtrx_tot, conf_mtrx):
    if conf_mtrx_tot.empty:
        conf_mtrx_tot = conf_mtrx.copy()
    else:
        conf_mtrx_tot = conf_mtrx_tot.add(conf_mtrx, fill_value=0)
    return conf_mtrx_tot


def postprocess(split_raw_json_path, definitive_folder, save_path):
    """Implements the postprocessing pipeline"""
    log = LoggerSingleton()
    log.name = "Postprocessing"

    dmg_labels = [i for i in range(1, len(LabelDict()))]
    os.makedirs(save_path, exist_ok=True)

    predicted_patches_folder_path = join(definitive_folder, "test_pred_masks")
    predicted_dataset = PredictedDataset(
        split_raw_json_path, predicted_patches_folder_path)

    px_conf_mtrx_tot = pd.DataFrame()
    px_multi_conf_mtrx_tot = pd.DataFrame()
    obj_conf_mtrx_tot = pd.DataFrame()
    obj_multi_conf_mtrx_tot = pd.DataFrame()
    for dis_id, tile_id, tile_dict, pred_mask in predicted_dataset:
        pred_out = os.path.join(save_path, f"{dis_id}_{tile_id}")
        os.makedirs(pred_out, exist_ok=True)
        px_conf_mtrx, px_multi_conf_mtrx =\
            pixel_analysis(tile_dict, pred_mask, dmg_labels, pred_out)
        px_conf_mtrx_tot = add_confusion_matrices(
            px_conf_mtrx_tot, px_conf_mtrx)
        px_multi_conf_mtrx_tot = add_confusion_matrices(
            px_multi_conf_mtrx_tot, px_multi_conf_mtrx)
        obj_conf_mtrx, obj_multi_conf_mtrx =\
            object_analysis(tile_dict, pred_mask, dmg_labels, pred_out)
        obj_conf_mtrx_tot = add_confusion_matrices(
            obj_conf_mtrx_tot, obj_conf_mtrx)
        obj_multi_conf_mtrx_tot = add_confusion_matrices(
            obj_multi_conf_mtrx_tot, obj_multi_conf_mtrx)
        make_figures(pred_out, dis_id, tile_id, tile_dict, pred_mask)

    metrics = MetricComputer.compute_eval_metrics(px_conf_mtrx_tot, dmg_labels)
    save_df(metrics, pred_out, "pixel_metrics")
    metrics = MetricComputer.compute_eval_metrics(
        obj_conf_mtrx_tot, dmg_labels)
    save_df(metrics, pred_out, "object_metrics")
    save_df(px_conf_mtrx_tot, save_path, "pixel_confusion_matrix")
    save_df(px_multi_conf_mtrx_tot, save_path, "pixel_multi_confusion_matrix")
    save_df(obj_conf_mtrx_tot, save_path, "obj_confusion_matrix")
    save_df(obj_multi_conf_mtrx_tot, save_path, "obj_multi_confusion_matrix")
