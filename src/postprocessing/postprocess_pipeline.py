import pandas as pd
import torch
from tqdm import tqdm
from utils.common.pathManager import FilePath
from utils.loggers.table_print import to_table
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


def save_df(metric_df: pd.DataFrame, pred_out: FilePath, file_name: str):
    """Save metrics in csv"""
    file = pred_out.join(file_name+".csv")
    metric_df.to_csv(path_or_buf=file, mode="w", index=False)
    file = pred_out.join(file_name+".tex")
    metric_df.to_latex(file)


def save_img(out_dir: FilePath, dis_id: str, tile_id: str, prefix: str,
             tile: torch.Tensor):
    img = tile.numpy()
    path = out_dir.join(f"{dis_id}_{tile_id}_{prefix}_disaster.png")
    LabelMaskVisualizer.save_arr_img(img, path)


def save_mask(out_dir: FilePath, dis_id: str, tile_id: str,
              pred_mask: torch.Tensor):
    pred_path = out_dir.join(f"{dis_id}_{tile_id}_pred_damage_mask.png")
    pred_img = LabelMaskVisualizer.draw_label_img(pred_mask)
    LabelMaskVisualizer.save_tensor_img(pred_img, pred_path)


def save_bbs(out_dir: FilePath, dis_id: str, tile_id: str, prefix: str,
             bbs_df: pd.DataFrame,):
    values = list(bbs_df.value_counts(subset=["label"]).items())
    table = pd.DataFrame([(val[0][0], val[1]) for val in values],
                         columns=["Level", "Count"])
    folder = out_dir.join(f"{prefix}_bbs")
    folder.create_folder()
    save_df(table, folder, f"{prefix}_table.csv")
    bbs_by_level_figures(dis_id, tile_id, bbs_df, folder)
    return table


def pixel_analysis(tile_dict: dict, pred_mask: torch.Tensor,
                   dmg_labels: list, pred_out: FilePath):
    conf = MatrixComputer.tile_px_conf_mtrx(tile_dict["bld_mask"],
                                            tile_dict["dmg_mask"],
                                            pred_mask, dmg_labels)
    save_df(conf, pred_out, "pixel_confusion_matrix")

    multi_conf = MatrixComputer.px_multiclass_conf_mtrx(
        tile_dict["dmg_mask"], pred_mask, dmg_labels)
    save_df(multi_conf, pred_out, "pixel_multiclass_confusion_matrix")

    metrics = MetricComputer.compute_eval_metrics(conf, dmg_labels)
    save_df(metrics, pred_out, "pixel_metrics")
    return conf, multi_conf


def object_analysis(tile_dict: dict, pred_mask: torch.Tensor,
                    dmg_labels: list, pred_out: FilePath):
    conf, multi_conf = MatrixComputer.\
        tile_obj_conf_matrices(tile_dict["dmg_mask"], pred_mask, dmg_labels)
    save_df(conf, pred_out, "object_confusion_matrix")
    save_df(multi_conf, pred_out, "object_multiclass_confusion_matrix")

    metrics = MetricComputer.compute_eval_metrics(conf, dmg_labels)
    save_df(metrics, pred_out, "object_metrics")
    return conf, multi_conf


def make_figures(out_dir: FilePath, dis_id: str, tile_id: str,
                 tile_dict: dict, pred_mask: torch.Tensor):
    log = LoggerSingleton()
    save_img(out_dir, dis_id, tile_id, "pre", tile_dict["pre_img"])
    save_img(out_dir, dis_id, tile_id, "post", tile_dict["post_img"])
    save_mask(out_dir, dis_id, tile_id, pred_mask)
    log.info(f"Tile images for {tile_id} saved.")
    gt_bbs_df = get_bbs_from_json(tile_dict["dmg_json"])
    gt_table = save_bbs(out_dir, dis_id, tile_id, "gt", gt_bbs_df)
    log.info(f"Tiles ground truth bbs for {tile_id} saved.")
    pd_bbs_df = get_bbs_from_mask(pred_mask)
    pd_table = save_bbs(out_dir, dis_id, tile_id, "pd", pd_bbs_df)
    log.info(f"Tiles predicted bbs for {tile_id} saved.")
    comparative_figure(dis_id, tile_id,  tile_dict["pre_img"],
                       tile_dict["post_img"], pred_mask,
                       gt_table, pd_table, out_dir)
    superposed_img(dis_id, tile_id,
                   tile_dict["pre_img"], tile_dict["post_img"], out_dir)
    log.info(f"Extra figures for {tile_id} saved.")


def add_confusion_matrices(conf_tot: pd.DataFrame, conf: pd.DataFrame):
    if conf_tot.empty:
        conf_tot = conf.copy()
    else:
        conf_tot = conf_tot.add(conf.iloc[:,1:], fill_value=0).astype(int)
    return conf_tot


def postprocess(split_json: FilePath, pred_dir: FilePath,
                out_dir: FilePath):
    """
    Implements the postprocessing pipeline.

    Args:
        tile_split_json_path (FilePath): Path to the JSON file containing tile
        splits.
        pred_dir (FilePath): Directory containing prediction masks.
        out_dir (FilePath): Output directory for saving postprocessed results.
    """
    log = LoggerSingleton("POSTPROCESSING", folder_path=out_dir)
    dmg_labels = list(range(1, len(LabelDict())))
    out_dir.create_folder()

    predicted_patches_folder_path = pred_dir.join("test_pred_masks")
    predicted_dataset = PredictedDataset(split_json,
                                         predicted_patches_folder_path)
    log.info(f"Post processing {len(predicted_dataset)} patches.")

    px_conf_tot, px_multi_conf_tot = pd.DataFrame(), pd.DataFrame()
    obj_conf_tot, obj_multi_conf_tot = pd.DataFrame(), pd.DataFrame()

    for dis_id, tile_id, tile_dict, pred_mask in tqdm(predicted_dataset,
                                                      desc="Predicted tile"):
        pred_out = out_dir.join(f"{dis_id}_{tile_id}")
        pred_out.create_folder()
        log.info(f"Computing pixel level metrics for {dis_id}-{tile_id}")
        px_conf, px_multi_conf = pixel_analysis(tile_dict, pred_mask,
                                                dmg_labels, pred_out)
        px_conf_tot = add_confusion_matrices(px_conf_tot, px_conf)
        px_multi_conf_tot = add_confusion_matrices(px_multi_conf_tot,
                                                   px_multi_conf)

        log.info(f"Computing pixel object level metrics for {dis_id}-{tile_id}")
        obj_conf, obj_multi_conf = object_analysis(tile_dict, pred_mask,
                                                   dmg_labels, pred_out)
        obj_conf_tot = add_confusion_matrices(obj_conf_tot, obj_conf)
        obj_multi_conf_tot = add_confusion_matrices(obj_multi_conf_tot,
                                                    obj_multi_conf)
        log.info(f"Generating figures for {dis_id}-{tile_id}")
        make_figures(pred_out, dis_id, tile_id, tile_dict, pred_mask)

    save_metrics_and_matrices(out_dir, px_conf_tot, px_multi_conf_tot,
                              obj_conf_tot, obj_multi_conf_tot, dmg_labels)


def save_metrics_and_matrices(out_dir: FilePath, px_conf_tot: pd.DataFrame,
                              px_multi_conf_tot: pd.DataFrame,
                              obj_conf_tot: pd.DataFrame,
                              obj_multi_conf_tot: pd.DataFrame,
                              dmg_labels: list):
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
    save_df(px_metrics, metric_dir, "pixel_metrics")
    log.info(to_table(curr_type="pixel", df=px_metrics,
                      odd=True, decim_digits=5))

    obj_metrics = MetricComputer.compute_eval_metrics(
        obj_conf_tot, dmg_labels)
    log = LoggerSingleton()
    log.info("Pixel Metrics")
    log.info(to_table(curr_type="object", df=obj_metrics,
                      odd=True, decim_digits=5))

    save_df(obj_metrics, metric_dir, "object_metrics")

    save_df(px_conf_tot, metric_dir, "pixel_confusion_matrix")
    save_df(px_multi_conf_tot, metric_dir, "pixel_multi_confusion_matrix")
    save_df(obj_conf_tot, metric_dir, "obj_confusion_matrix")
    save_df(obj_multi_conf_tot, metric_dir, "obj_multi_confusion_matrix")
