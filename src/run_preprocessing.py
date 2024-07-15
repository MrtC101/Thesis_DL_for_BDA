# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys
from os.path import join
# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
from utils.common.timeManager import measure_time
from utils.common.pathManager import FilePath
from utils.loggers.console_logger import LoggerSingleton
from preprocessing.preprocessing_pipeline import preprocess

os.environ["PROJ_PATH"] = "/home/mrtc101/Desktop/tesina/repo/hiper_siames"
os.environ["SRC_PATH"] = join(os.environ["PROJ_PATH"], "src")
os.environ["DATA_PATH"] = join(os.environ["PROJ_PATH"], "data")
os.environ["OUT_PATH"] = join(os.environ["PROJ_PATH"], "out")

if __name__ == "__main__":
    out_path = FilePath(os.environ["OUT_PATH"]).join("preprocessing")
    log = LoggerSingleton("PREPROCESSING", folder_path=out_path)

    DISASTERS_OF_INTEREST = (
        # 'guatemala-volcano',
        # 'hurricane-florence',
        # 'midwest-flooding',
        # 'socal-fire',
        # 'hurricane-matthew',
        # 'hurricane-harvey',
        # 'hurricane-michael',
        # 'santa-rosa-wildfire',
        # 'palu-tsunami',
        'mexico-earthquake',
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
    disaster_num = 20

    tile_splits_json_path, aug_splits_json_path, \
        patch_split_json_path, mean_std_json_path = \
        measure_time(preprocess, disaster_num, DISASTERS_OF_INTEREST)

    paths = {
        "tile_splits_json_path": tile_splits_json_path,
        "aug_splits_json_path": aug_splits_json_path,
        "patch_split_json_path": patch_split_json_path,
        "mean_std_json_path": mean_std_json_path
    }
    out_path.join("data_paths.json").save_json(paths)
