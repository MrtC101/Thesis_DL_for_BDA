# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from preprocessing.create_params.create_params import create_params
from utils.common.timeManager import measure_time
from utils.common.pathManager import FilePath
from preprocessing.preprocessing_pipeline import preprocess

if __name__ == "__main__":
    out_path = FilePath(os.environ["OUT_PATH"])
    paths = measure_time(preprocess)
    out_path.join("data_paths.json").save_json(paths)
    # LOAD PARAMETERS
    params = FilePath(os.environ["EXP_PATH"]) \
        .join("params.yml").read_yaml()
    create_params(params)
