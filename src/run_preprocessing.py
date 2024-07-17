# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys
from os.path import join
os.environ["PROJ_PATH"] = "/home/mrtc101/Desktop/tesina/repo/hiper_siames"
os.environ["SRC_PATH"] = join(os.environ["PROJ_PATH"], "src")
os.environ["DATA_PATH"] = join(os.environ["PROJ_PATH"], "data")
os.environ["OUT_PATH"] = join(os.environ["PROJ_PATH"], "out")

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.timeManager import measure_time
from utils.common.pathManager import FilePath
from preprocessing.preprocessing_pipeline import preprocess

"""
disasters_of_interest = (
        # 'guatemala-volcano',
        # 'hurricane-florence',
        # 'midwest-flooding',
        # 'socal-fire',
        # 'hurricane-matthew',
        # 'hurricane-harvey',
        # 'hurricane-michael',
        # 'santa-rosa-wildfire',
        # 'palu-tsunami',
        # 'mexico-earthquake',
        # 'joplin-tornado',
        # 'lower-puna-volcano',
        # 'moore-tornado',
        # 'nepal-flooding',
        # 'pinery-bushfire',
        # 'portugal-wildfire',
        # 'sunda-tsunami',
        # 'tuscaloosa-tornado',
        # 'woolsey-fire'
    )
"""
if __name__ == "__main__":
    out_path = FilePath(os.environ["OUT_PATH"])
    disasters_of_interest: tuple = (
        'mexico-earthquake',
    )
    disaster_num = 10 * 100000
    num_aug = 5

    paths = measure_time(preprocess, disaster_num, num_aug,
                         disasters_of_interest, out_path)
    (tile_splits_json_path, patch_split_json_path,
     aug_tile_split_json_path, aug_patch_split_json_path,
     mean_std_json_path) = paths

    paths = {
        "tile_splits_json_path": tile_splits_json_path,
        "aug_tile_split_json_path": aug_tile_split_json_path,
        "aug_patch_split_json_path": aug_patch_split_json_path,
        "patch_split_json_path": patch_split_json_path,
        "mean_std_json_path": mean_std_json_path
    }
    out_path.join("data_paths.json").save_json(paths)
