import os
import sys

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from training.run_on_test_pipeline import inference_on_test
from training.fold_resume.fold_metrics import get_best_config
from utils.common.pathManager import FilePath
from utils.common.timeManager import measure_time

if __name__ == "__main__":
    out_path = FilePath(os.environ["OUT_PATH"])
    paths = out_path.join("data_paths.json").read_json()
    aug_patch_split_json_path = FilePath(paths['patch_split_json_path'])
    mean_std_json_path = FilePath(paths['mean_std_json_path'])

    param_dict = out_path.join("conf_list.json").read_json()
    if(len(param_dict) == 1):
        best_config = param_dict[0]
    else:
        best_config = get_best_config(out_path, param_dict)
    paths_dict = {
        "split_json": aug_patch_split_json_path,
        "mean_json": mean_std_json_path,
        "out_dir": out_path.join("definitive_model"),
    }

    measure_time(inference_on_test, best_config[1], paths_dict)
