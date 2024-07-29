# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
from utils.common.pathManager import FilePath
from utils.common.timeManager import measure_time
from training.hiper_search_pipeline import parameter_search


if __name__ == "__main__":

    out_path = FilePath(os.environ["OUT_PATH"])
    paths = out_path.join("data_paths.json").read_json()

    paths_dict = {
        "split_json": FilePath(paths['patch_split_json_path']),
        "mean_json": FilePath(paths['mean_std_json_path']),
        "out_dir": out_path,
    }

    param_list = out_path.join("conf_list.json").read_json()
    conf_num = int(os.environ["CONF_NUM"])
    current_params = tuple(param_list[conf_num])

    folds = current_params[1]["folds"]

    measure_time(parameter_search, folds, current_params, paths_dict)
