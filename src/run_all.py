# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
from postprocessing.postprocess_pipeline import postprocess
from preprocessing.preprocessing_pipeline import preprocess
from utils.common.pathManager import FilePath
from utils.common.timeManager import measure_time
from training.train_pipeline import train_definitive
from training.hiper_search_pipeline import parameter_search
import os
import sys
from os.path import join

os.environ["PROJ_PATH"] = "/home/mcogo/scratch"
os.environ["SRC_PATH"] = join(os.environ["PROJ_PATH"], "src")
os.environ["DATA_PATH"] = join(os.environ["PROJ_PATH"], "data")
os.environ["OUT_PATH"] = join(os.environ["PROJ_PATH"], "out")

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))


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
        'nepal-flooding',
        'midwest-flooding',
        'santa-rosa-wildfire',
        'portugal-wildfire',
        'socal-fire',
        'woolsey-fire',
        'pinery-bushfire',
    )
    disaster_num = 10 * 100000
    num_aug = 0

    paths = measure_time(preprocess, disaster_num, num_aug,
                         disasters_of_interest, out_path)
    (tile_splits_json_path, patch_split_json_path,
     aug_tile_split_json_path, aug_patch_split_json_path,
     mean_std_json_path) = paths

    # Configuration dictionaries for paths used during model training
    weights_config = {
        'weights_seg': [1, 15],
        'weights_damage': [1, 35, 70, 150, 120],
        'weights_loss': [0, 0, 1],
    }
    hardware_config = {
        'torch_threads': 64,
        'torch_op_threads': 64,
        'batch_workers': 0,
        'new_optimizer': False,
    }
    visual_config = {
        'num_chips_to_viz': 1,
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
        'init_learning_rate': [0.0005, 0.001, 0.01, 0.1],
        'tot_epochs': [100, 300, 500],
        'batch_size': [16, 32, 64]
    }
    best_config = measure_time(parameter_search, 10, hyperparameter_config,
                               configs, paths_dict)

    paths_dict["split_json"] = aug_patch_split_json_path
    paths_dict['out_dir'] = out_path.join("definitive_model")
    definitive_acc_score = measure_time(train_definitive, best_config,
                                        paths_dict)

    paths = {}
    paths["split_json"] = aug_tile_split_json_path
    paths['pred_dir'] = paths_dict['out_dir']
    paths['out_dir'] = out_path.join("postprocessing")

    measure_time(postprocess, **paths)
