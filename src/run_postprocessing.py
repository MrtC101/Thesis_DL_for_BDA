# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys
from os.path import join
#os.environ["PROJ_PATH"] = "/home/mrtc101/Desktop/tesina/repo/hiper_siames"
#os.environ["SRC_PATH"] = join(os.environ["PROJ_PATH"], "src")
#os.environ["DATA_PATH"] = join(os.environ["PROJ_PATH"], "data")
#os.environ["OUT_PATH"] = join(os.environ["PROJ_PATH"], "out")

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.pathManager import FilePath
from utils.common.timeManager import measure_time
from postprocessing.postprocess_pipeline import postprocess

if __name__ == "__main__":
    out_path = FilePath(os.environ["OUT_PATH"])
    paths = out_path.join("data_paths.json").read_json()
    aug_tile_split_json_path = FilePath(paths['aug_tile_split_json_path'])

    paths = {}
    paths["split_json"] = aug_tile_split_json_path
    paths['pred_dir'] = out_path.join("definitive_model")
    paths['out_dir'] = out_path.join("postprocessing")

    measure_time(postprocess, **paths)
