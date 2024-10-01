# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.common.defaultDictFactory import nested_defaultdict
from utils.common.pathManager import FilePath
from utils.loggers.console_logger import LoggerSingleton
from utils.datasets.raw_datasets import TileDataset
from utils.visualization.label_to_color import LabelDict
from cv2 import imread

log = LoggerSingleton()


def compute_mean_stddev(pre_img: np.ndarray, post_img: np.ndarray) -> dict:
    """Computes the mean and standard deviation for each channel from pre
       and post disaster images."""

    def calculate_stats(img: np.ndarray) -> dict:
        norm_img = img / 255.0
        mean = {channel: float(norm_img[:, :, i].mean())
                for i, channel in enumerate("RGB")}
        stdv = {channel: float(norm_img[:, :, i].std())
                for i, channel in enumerate("RGB")}
        return {"mean": mean, "stdv": stdv}

    return {
        "pre": calculate_stats(pre_img),
        "post": calculate_stats(post_img)
    }


def create_data_dicts(splits_json_path: FilePath, out_path: FilePath) -> str:
    """Creates three JSON files and stores them in a folder named \
        `dataset_statistics` inside the specified `out_path`.

    Args:
        splits_json_path: Path to the JSON file that stores the dataset splits.
        out_path: Path to the folder where the 3 new JSON files will be saved.

    Returns:
        str: The path to the new `dataset_statistics` folder.

    Files created:
        - `all_tiles_count_area.json` this file stores the number of polygons
        with each damage class present inside a mask for each tile inside the
        xBD dataset folder.
        - `all_tiles_count_area_by_disaster.json` this file stores the total
        count but by disaster.
        - `all_tiles_mean_stddev.json` this file stores the mean and standard
          deviation of each color channel inside a all disaster tile image from
          each dataset split.

    Example:
        >>> create_data_dicts("data/xBD/splits/raw_splits.json","data/xBD/raw")
    """
    out_path.must_be_dir()
    dicts_path = out_path.join("dataset_statistics")
    dicts_path.create_folder()
    splits = list(splits_json_path.read_json().keys())

    mean = nested_defaultdict(2, dict)
    for split_name in tqdm(splits):
        dataset = TileDataset(split_name=split_name,
                              splits_json_path=splits_json_path)
        log.info(f'Counting {split_name} subset with length: {len(dataset)}')
        for dis_id, tile_id, data in tqdm(iter(dataset), total=len(dataset)):
            mean[dis_id][tile_id] = compute_mean_stddev(data["pre_img"],
                                                        data["post_img"])
    mean_path = dicts_path.join("all_tiles_mean_stddev.json")
    mean_path.save_json(dict(mean))
    return dicts_path
