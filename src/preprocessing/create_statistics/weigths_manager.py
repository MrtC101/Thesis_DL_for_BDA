import pandas as pd
from cv2 import imread
import numpy as np
from sklearn.model_selection import ParameterGrid
from utils.common.pathManager import FilePath
from utils.loggers.console_logger import LoggerSingleton
from utils.visualization.label_to_color import LabelDict


def _count_pixels(mask_path: FilePath) -> list:
    """Count the number of pixels of each class present in the image"""
    row = np.zeros(6, np.uint16)
    count_tup = np.unique(imread(mask_path)[:, :, 0], return_counts=True)
    for label, count in zip(count_tup[0], count_tup[1]):
        row[label] = count
    return list(row)


def compute_pixel_weights(split_json_path: FilePath):
    """Computes the weights for each class counting all labeled pixels from the train split
    of the xBD dataset.
    Args:
        split_json_path (FilePath): Path to the JSON file where are represented all splits
            and the path of each tile from the xBD dataset.
    Returns:
        dict: A dictionary with the corresponding weights for classification and segmentation.
    """
    log = LoggerSingleton()

    # Counting pixels from each tile's damage mask.
    rows = []
    for split_id, dis_dict in split_json_path.read_json().items():
        for dis_id, tile_dict in dis_dict.items():
            for tile_id in tile_dict.keys():
                dmg_mask_path = FilePath(tile_dict[tile_id]["post"]["mask"])
                px_count = _count_pixels(dmg_mask_path)
                rows.append([split_id, dis_id, tile_id] + px_count)

    # Building a DataFrame of tiles with each corresponding count
    index_cols = ["split_id", "dis_id", "tile_id"]
    labels_list = list(LabelDict().labels.keys())
    bld_per_tile = pd.DataFrame(rows, columns=index_cols + labels_list).set_index(index_cols)
    count_per_class_dmg = bld_per_tile.sum()

    # Computing weights for damage labels ignoring "unclassified"
    dmg_weights = count_per_class_dmg.sum() / count_per_class_dmg
    dmg_weights[(count_per_class_dmg <= 0)] = 0.0
    dmg_weights: pd.Series = dmg_weights.loc[labels_list[0:5]]
    dmg_w_list = [round(dmg_weights.loc[label], 0) for label in labels_list[0:5]]
    log.info(dmg_weights)

    # Computing weights for segmentation labels summing all the others different from "background"
    seg_labels = ["background", "building"]
    count_per_class_seg = pd.Series(data=[
        count_per_class_dmg.loc["background"],
        count_per_class_dmg.loc[labels_list[1:5]].sum()
    ], index=seg_labels)
    seg_weights: pd.Series = count_per_class_seg.sum() / count_per_class_seg
    seg_w_list = [round(seg_weights.loc[label], 0) for label in seg_labels]
    log.info(seg_weights)

    return {"seg": seg_w_list, "dmg": dmg_w_list}


def create_configs(params_path: FilePath, weights: dict) -> dict:
    """Create a list of configuration dictionaries for hyperparameter optimization.

    Args:
        configs (dict): Base configuration dictionary with default settings.

    Returns:
        list[dict]: List of configuration dictionaries with different
          hyperparameter combinations.
    """
    params_dict = params_path.read_yaml()

    configs = {**params_dict['train'], **params_dict['visual'],
               **params_dict['preprocessing'], **params_dict['weights']}
    configs["weights_dmg"] = weights["dmg"]
    configs["weights_seg"] = weights["seg"]
    param_combinations = list(ParameterGrid(params_dict['hyperparameter']))
    configs = {i: {**configs, **params} for i, params in enumerate(param_combinations)}
    return configs
