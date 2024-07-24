# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys
# os.environ["PROJ_PATH"] = "/home/mrtc101/Desktop/tesina/repo/hiper_siames"
# os.environ["OUT_PATH"] = join(os.environ["PROJ_PATH"], "out")

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
from training.hiper_search_pipeline import create_params
from utils.common.pathManager import FilePath


if __name__ == "__main__":

    out_path = FilePath(os.environ["OUT_PATH"])
    params_json = out_path.join("param_list.json")
    weights_dict = out_path.join("train_weights.json").read_json()
    weights = [1]  # w for label 0
    lab_w = [round(w) for w in weights_dict.values()]
    lab_w.reverse()
    weights.extend(lab_w)  # weights for all classes
    hyperparameter_config = {
        'init_learning_rate': [0.1],
        'tot_epochs': [1],
        'batch_size': [20]
    }
    # Configuration dictionaries for paths used during model training
    weights_config = {
        'weights_seg': [1, 15],
        'weights_damage': weights,
        'weights_loss': [0, 0, 1],
    }
    hardware_config = {
        'torch_threads': 64,
        'torch_op_threads': 64,
        'batch_workers': 0,
        'new_optimizer': False,
        "checkpoint": True
    }
    visual_config = {
        'num_chips_to_viz': 1,
        'labels_dmg': [0, 1, 2, 3, 4],
        'labels_bld': [1],  # Do not include 0 because it is binary
    }
    configs = {**weights_config, **hardware_config, **visual_config}
    param_list = create_params(hyperparameter_config, configs)
    params_json.save_json({"param_list": param_list})
