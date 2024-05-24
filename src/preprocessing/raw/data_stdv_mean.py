# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Modificaciones (c) 2024 Martín Cogo Belver.
# Martín Cogo Belver has rights reserved over this modifications.
#
# Modification Notes:
# - Documentation added with docstrings for code clarity.
# - Re-implementation of methods to enhance readability and efficiency.
# - Re-implementation of features for improved functionality.
# - Changes in the logic of implementation for better performance.
# - Bug fixes in the code.
#
# See the LICENSE file in the root directory of this project for the full text of the MIT License.import os
import sys
import numpy as np
from utils.common.files import dump_json, is_dir
from utils.datasets.raw_datasets import RawDataset
from collections import defaultdict
from shapely import wkt
import argparse
from tqdm import tqdm
from utils.common.logger import LoggerSingleton, TqdmToLog

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
log = LoggerSingleton()


def compute_mean_stddev(pre_img: np.array, post_img: np.array) -> dict:
    """Computes the mean and standard deviation from each tile
    from each disaster.
    """
    mean = {}
    for time, img in zip(["pre", "post"], [pre_img, post_img]):
        mean[time] = {}
        norm_img = img / 255.0
        mean[time]["mean"] = {
            "R": float(norm_img[:, :, 0].mean()),
            "G": float(norm_img[:, :, 1].mean()),
            "B": float(norm_img[:, :, 2].mean()),
        }
        mean[time]["stdv"] = {
            "R": float(norm_img[:, :, 0].std()),
            "G": float(norm_img[:, :, 1].std()),
            "B": float(norm_img[:, :, 2].std()),
        }
    return mean


def count_buildings(pre_json: dict, post_json: dict) -> dict:
    """Counts buildings' class and area from each tile from each disaster."""
    count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for time, file in zip(["pre", "post"], [pre_json, post_json]):
        for coord in file['features']['xy']:
            feature_type = coord['properties']['feature_type']
            if feature_type != 'building':
                count[time]["not_building"][feature_type] += 1

            damage_class = coord['properties'].get('subtype', 'no-subtype')

            feat_shape = wkt.loads(coord['wkt'])

            if ("bld_area" not in count[time].keys()):
                count[time]["bld_area"] = defaultdict(float)

            count[time]["bld_area"][damage_class] += feat_shape.area
            count[time]["bld_class"][damage_class] += 1

    return count


def count_by_disaster(count: dict) -> dict:
    """Sums the bld_area and bld_class for all tiles from each disaster."""
    c_by_d = {}
    for zone_id, zone in tqdm(count.items()):
        c_by_d[zone_id] = {
            "post": {
                "bld_area": defaultdict(int),
                "bld_class": defaultdict(int)
            },
            "pre": {
                "bld_area": defaultdict(int),
                "bld_class": defaultdict(int)
            }
        }
        for tile_id, tile in zone.items():
            for time_id, time in tile.items():
                for m_id, mesure in time.items():
                    for val_id, value in mesure.items():
                        c_by_d[zone_id][time_id][m_id][val_id] += value
    return c_by_d


def create_data_dicts(splits_json_path: str, out_path: str) -> str:
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
    log.name = "Compute Data Statistics"
    is_dir(out_path)
    dicts_path = os.path.join(out_path, "dataset_statistics")
    os.makedirs(dicts_path, exist_ok=True)

    mean = defaultdict(lambda: {})
    count = defaultdict(lambda: {})

    for split_name in ["train", "val"]:
        dataset = RawDataset(split_name=split_name,
                             splits_json_path=splits_json_path)
        log.info(f'Counting {split_name} subset with length: {len(dataset)}')
        psb = tqdm(iter(dataset), total=len(dataset))
        for dis_id, tile_id, data in psb:
            mean[dis_id][tile_id] = compute_mean_stddev(data["pre_img"], data["post_img"])
            count[dis_id][tile_id] = count_buildings(data["pre_json"], data["post_json"])

    mean_path = os.path.join(dicts_path, "all_tiles_mean_stddev.json")
    count_path = os.path.join(dicts_path, "all_tiles_count_area.json")
    dump_json(mean_path, dict(mean))
    dump_json(count_path, dict(count))

    log.info('Total counting by each disaster.')
    c_by_d = count_by_disaster(count=count)
    mean_disaster_path = os.path.join(dicts_path, "all_tiles_count_area_by_disaster.json")
    dump_json(mean_disaster_path, c_by_d)
    return dicts_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This method creates 3 JSON files and stores them inside \
            a folder `dataset_statistics` that will be created inside the \
            `out_path`.")
    parser.add_argument(
        'split_json_path',
        type=str,
        help=('Path to the JSON file that stores the dataset splits.')
    )
    parser.add_argument(
        'out_path',
        type=str,
        help=('Path to the folder where the 3 new JSON files will be saved.')
    )
    args = parser.parse_args()
    create_data_dicts(args.split_json_path, args.out_path)
