# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from training.cross_validation_pipeline import k_cross_validation
from utils.common.pathManager import FilePath


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
    config = param_list[str(conf_num)]

    paths_dict['out_dir'] = FilePath(paths_dict['out_dir']).join(f'config-{conf_num}')
    paths_dict['out_dir'].create_folder()
    config['configuration_num'] = conf_num
    k_cross_validation(config["folds"], config, paths_dict)
