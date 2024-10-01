import pandas as pd
import torch
from tqdm import tqdm
from postprocessing.file_creation.post_file_save import save_bbs, save_df, save_img, save_mask, \
    save_metrics_and_matrices
from utils.common.pathManager import FilePath
from utils.common.timeManager import measure_time
from utils.visualization.label_to_color import LabelDict
from utils.metrics.matrix_computer import MatrixComputer
from utils.metrics.metric_manager import MetricComputer
from utils.loggers.console_logger import LoggerSingleton
from utils.datasets.predicted_dataset import PredictedDataset
from postprocessing.plots.plot_results import comparative_figure, superposed_img
from postprocessing.bbs.bounding_boxes import get_bbs_from_json, get_bbs_from_mask


def pixel_analysis(tile_dict: dict, pred_mask: torch.Tensor, dmg_labels: list, pred_out: FilePath):
    """Pixel level evaluation of predicted masks with ground truth mask.
    Args:
        tile_dict: file paths dictionary for the ground truth tile.
        pred_mask: predicted damage mask of 1024x1024
        dmg_labels: list of labels numbers to evaluate.
        pred_out: path to the directory where evaluation metric should be stored.
    Returns:
        pd.Dataframe: Current predicted mask confusion matrix.
        pd.Dataframe: Current predicted mask prediction table.
    """
    conf = MatrixComputer.tile_px_conf_mtrx(tile_dict["bld_mask"], tile_dict["dmg_mask"],
                                            pred_mask, dmg_labels)
    save_df(conf, pred_out, "pixel_confusion_matrix")

    multi_conf = MatrixComputer.px_multiclass_conf_mtrx(tile_dict["dmg_mask"], pred_mask,
                                                        dmg_labels)
    save_df(multi_conf, pred_out, "pixel_multiclass_confusion_matrix")

    metrics = MetricComputer.compute_eval_metrics(conf, dmg_labels)
    save_df(metrics, pred_out, "pixel_metrics")
    return conf, multi_conf


def object_analysis(tile_dict: dict, pred_mask: torch.Tensor, dmg_labels: list,
                    pred_out: FilePath) -> tuple:
    """Object level evaluation of predicted mask with ground truth mask.
    Args:
        tile_dict: file paths dictionary for the ground truth tile.
        pred_mask: predicted damage mask of 1024x1024
        dmg_labels: list of labels numbers to evaluate.
        pred_out: path to the directory where evaluation metric should be stored.
    Returns:
        pd.Dataframe: Current predicted mask confusion matrix.
        pd.Dataframe: Current predicted mask prediction table.
    """

    conf, multi_conf = MatrixComputer.\
        tile_obj_conf_matrices(tile_dict["dmg_mask"], pred_mask, dmg_labels)
    save_df(conf, pred_out, "object_confusion_matrix")
    save_df(multi_conf, pred_out, "object_multiclass_confusion_matrix")

    metrics = MetricComputer.compute_eval_metrics(conf, dmg_labels)
    save_df(metrics, pred_out, "object_metrics")
    return conf, multi_conf


def make_figures(out_dir: FilePath, dis_id: str, tile_id: str, tile_dict: dict,
                 pred_mask: torch.Tensor) -> None:
    """
            Figures:
            - Pre and post images
            - Predicted damage mask image
            - Damage and post superposed image
            - Images with building count tables Figure
            - A folder with all corresponding bounding boxes for each class.
    Args:
        out_dir: Path to the directory to output all figures
        dis_id: Disaster label identifier from xBD dataset.
        tile_id: Tile label identifier from xBD dataset.
        tile_dict: dictionary with all files for the current tile_di.
        pred_mask: damage mask predicted
    """
    log = LoggerSingleton()
    # Saves each image
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
    comparative_figure(dis_id, tile_id,  tile_dict["pre_img"], tile_dict["post_img"], pred_mask,
                       gt_table, pd_table, out_dir)
    superposed_img(dis_id, tile_id, tile_dict["pre_img"], tile_dict["post_img"], out_dir)
    log.info(f"Extra figures for {tile_id} saved.")


def add_confusion_matrices(conf_tot: pd.DataFrame, conf: pd.DataFrame):
    """Adds two `pd.Dataframe` with a determined structure of columns"""
    if conf_tot.empty:
        conf_tot = conf.copy()
    else:
        conf_tot = conf_tot.add(conf.iloc[:, 1:], fill_value=0).astype(int)
    return conf_tot

@measure_time
def postprocess(split_json: FilePath, pred_dir: FilePath,
                out_dir: FilePath):
    """
    Implements the postprocessing pipeline.
        1. Computes pixel level evaluation metrics for all predicted damaged masks of 1024x1024
        2. Compute object level evaluation metrics for all predicted damage masks of 1024x1024
        3. Generates figures and stores the inside `predicted dir` for each predicted mask.

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
    predicted_dataset = PredictedDataset(split_json, predicted_patches_folder_path)
    log.info(f"Post processing {len(predicted_dataset)} patches.")

    px_conf_tot, px_multi_conf_tot = pd.DataFrame(), pd.DataFrame()
    obj_conf_tot, obj_multi_conf_tot = pd.DataFrame(), pd.DataFrame()

    for dis_id, tile_id, tile_dict, pred_mask in tqdm(predicted_dataset, desc="Predicted tile"):
        pred_out = out_dir.join(f"{dis_id}_{tile_id}")
        pred_out.create_folder()

        log.info(f"Computing pixel level metrics for {dis_id}-{tile_id}")
        px_conf, px_multi_conf = pixel_analysis(tile_dict, pred_mask, dmg_labels, pred_out)
        px_conf_tot = add_confusion_matrices(px_conf_tot, px_conf)
        px_multi_conf_tot = add_confusion_matrices(px_multi_conf_tot, px_multi_conf)

        log.info(f"Computing pixel object level metrics for {dis_id}-{tile_id}")
        obj_conf, obj_multi_conf = object_analysis(tile_dict, pred_mask, dmg_labels, pred_out)
        obj_conf_tot = add_confusion_matrices(obj_conf_tot, obj_conf)
        obj_multi_conf_tot = add_confusion_matrices(obj_multi_conf_tot, obj_multi_conf)

        log.info(f"Generating figures for {dis_id}-{tile_id}")
        make_figures(pred_out, dis_id, tile_id, tile_dict, pred_mask)

    save_metrics_and_matrices(out_dir, px_conf_tot, px_multi_conf_tot, obj_conf_tot,
                              obj_multi_conf_tot, dmg_labels)
