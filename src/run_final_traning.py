"""Run the training step for the final model with the hole training split and the configuration
with the highest harmonic f1 score over validation set."""
import os
import sys

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from training.model_train.utils import get_best_config
from utils.common.pathManager import FilePath
from training.train_final_pipeline import train_final_model

if __name__ == "__main__":
    out_path = FilePath(os.environ["OUT_PATH"])
    paths = out_path.join("data_paths.json").read_json()
    param_dict = out_path.join("conf_list.json").read_json()

    best_config = get_best_config(out_path, param_dict)

    paths_dict = {
        "split_json":  FilePath(paths['patch_split_json_path']),
        "mean_json": FilePath(paths['mean_std_json_path']),
        "out_dir": out_path.join("final_model")
    }
    train_final_model(best_config, paths_dict)
