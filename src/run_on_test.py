# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from training.model_train.utils import get_best_config
from utils.common.pathManager import FilePath
from training.model_train.run_on_test_pipeline import inference_on_test
from training.model_train.utils import TrainDataLoader, set_threads
from utils.loggers.console_logger import LoggerSingleton
from utils.datasets.train_dataset import TrainDataset


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
    
    # Logger
    out_dir = FilePath(paths_dict['out_dir'])
    log_out = out_dir.join("console_logs")
    log = LoggerSingleton("FINAL MODEL", folder_path=log_out)
    log.info(f"Using best configuration with number {best_config[0]}")
    configs = best_config[1]
    set_threads()
    configs['tot_epochs'] = configs['final_epochs']
    xBD_test = TrainDataset('test', paths_dict['split_json'], paths_dict['mean_json'])
    test_loader = TrainDataLoader(xBD_test, batch_size=configs['batch_size'])
    log.info(f'xBD_disaster_dataset TEST length: {len(xBD_test)}')
    inference_on_test(configs, paths_dict, test_loader)