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
from utils.common.pathManager import FilePath
from utils.loggers.console_logger import LoggerSingleton

if __name__ == "__main__":
    out_path = FilePath(os.environ["OUT_PATH"]).join("preprocessing")
    log = LoggerSingleton("PREPROCESSING", folder_path=out_path)
