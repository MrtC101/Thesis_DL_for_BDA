# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys
#os.environ["PROJ_PATH"] = "/home/mrtc101/Desktop/tesina/repo/hiper_siames"
#os.environ["SRC_PATH"] = join(os.environ["PROJ_PATH"], "src")
#os.environ["DATA_PATH"] = join(os.environ["PROJ_PATH"], "data")
#os.environ["OUT_PATH"] = join(os.environ["PROJ_PATH"], "out")

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
from utils.common.pathManager import FilePath
from utils.common.timeManager import measure_time
from training.hiper_search_pipeline import parameter_search


if __name__ == "__main__":

    out_path = FilePath(os.environ["OUT_PATH"])
    paths = out_path.join("data_paths.json").read_json()
    tile_splits_json_path = FilePath(paths['tile_splits_json_path'])
    patch_split_json_path = FilePath(paths['patch_split_json_path'])
    aug_tile_split_json_path = FilePath(paths['aug_tile_split_json_path'])
    aug_patch_split_json_path = FilePath(paths['aug_patch_split_json_path'])
    mean_std_json_path = FilePath(paths['mean_std_json_path'])

    paths_dict = {
        "split_json": patch_split_json_path,
        "mean_json": mean_std_json_path,
        "out_dir": out_path,
    }

    param_list = out_path.join("param_list.json").read_json()["param_list"]
    conf_num = int(os.environ["CONF_NUM"])
    current_params = param_list[conf_num: conf_num+1]
    measure_time(parameter_search, 5, current_params, paths_dict)
