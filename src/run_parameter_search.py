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

from training.hiper_search_pipeline import parameter_search
from utils.common.timeManager import measure_time
from utils.common.pathManager import FilePath

if __name__ == "__main__":

    out_path = FilePath(os.environ["OUT_PATH"])
    paths = out_path.join("data_paths.json").read_json()
    tile_splits_json_path = FilePath(paths['tile_splits_json_path'])
    patch_split_json_path = FilePath(paths['patch_split_json_path'])
    aug_tile_split_json_path = FilePath(paths['aug_tile_split_json_path'])
    aug_patch_split_json_path = FilePath(paths['aug_patch_split_json_path'])
    mean_std_json_path = FilePath(paths['mean_std_json_path'])


    # Configuration dictionaries for paths used during model training
    weights_config = {
        'weights_seg': [1, 15],
        'weights_damage': [1, 35, 70, 150, 120],
        'weights_loss': [0, 0, 1],
    }
    hardware_config = {
        'torch_threads': 12,
        'torch_op_threads': 12,
        'batch_workers': 0,
        'new_optimizer': False,
    }
    visual_config = {
        'num_chips_to_viz': 2,
        'labels_dmg': [0, 1, 2, 3, 4],
        'labels_bld': [1],  # Do not include 0 because it is binary
    }
    configs = {**weights_config, **hardware_config, **visual_config}

    paths_dict = {
        "split_json": patch_split_json_path,
        "mean_json": mean_std_json_path,
        "out_dir": out_path,
        "checkpoint": None
    }
    hyperparameter_config = {
        'init_learning_rate': [0.0005],
        'tot_epochs': [1],
        'batch_size': [25]
    }
    best_config = measure_time(parameter_search, 2, hyperparameter_config,
                               configs, paths_dict)

    out_path.join("best_params.json").save_json(best_config)
