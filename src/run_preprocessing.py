# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.pathManager import FilePath
from preprocessing.preprocessing_pipeline import preprocess

if __name__ == "__main__":
    out_path = FilePath(os.environ["OUT_PATH"])
    exp_path = FilePath(os.environ["EXP_PATH"])
    xbd_path = FilePath(os.environ["XBD_PATH"])
    data_path = FilePath(os.environ["DATA_PATH"])
    paths = preprocess(out_path, exp_path, xbd_path, data_path)
    out_path.join("data_paths.json").save_json(paths)
