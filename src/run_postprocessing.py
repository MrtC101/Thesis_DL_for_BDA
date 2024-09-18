import os
import sys

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.pathManager import FilePath
from utils.common.timeManager import measure_time
from postprocessing.postprocess_pipeline import postprocess

if __name__ == "__main__":
    out_path = FilePath(os.environ["OUT_PATH"])
    paths_dict = out_path.join("data_paths.json").read_json()

    args = {}
    args["split_json"] = FilePath(paths_dict['tile_splits_json_path'])
    args['pred_dir'] = out_path.join("definitive_model")
    args['out_dir'] = out_path.join("postprocessing")

    measure_time(postprocess, **args)
