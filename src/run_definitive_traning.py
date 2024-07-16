# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
from utils.loggers.console_logger import LoggerSingleton
from utils.common.pathManager import FilePath
from train.train import train_definitive
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


def train_definitive_model(configuration: dict) -> float:
    out_path = FilePath(os.environ["OUT_PATH"])
    s_path = out_path.join("hyper_parameter_search")
    log = LoggerSingleton("HYPERPARAMETER_SEARCH", folder_path=s_path)
    # Train definitive model
    pred_path = out_path.join("definitive_model")
    definitive_acc_score = train_definitive(pred_path, configuration)
    log.info(f"Accuracy for the final model : {definitive_acc_score}")
    return definitive_acc_score
