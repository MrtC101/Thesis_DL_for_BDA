import math
import os
import sys
from os.path import join
import pandas as pd
import numpy as np
import pandas as pd
from shapely.wkt import loads

from utils.metrics.metric_manager import MetricManager

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.logger import LoggerSingleton
from utils.datasets.predicted_dataset import PredictedDataset
from postprocessing.ploting.plotResults import save_results


def get_bbs_form_json(label_dict : dict) -> pd.DataFrame:
    """Create a pandas Dataframe with bounding boxes from json."""
    buildings_list = label_dict['features']['xy']
    bbs_list = []
    for build in buildings_list:
        type = build["properties"]["feature_type"]
        if(type == "building"):
            label = build["properties"]["subtype"]
            uid = build["properties"]["uid"]
            poly = loads(build["wkt"])
            x_e, y_e = poly.exterior.xy
            x1, y1 = math.floor(np.min(x_e)), math.floor(np.min(y_e))
            x2, y2 = math.floor(np.max(x_e)), math.floor(np.max(y_e))            
            x = max(x1,0)
            y = max(y1,0)
            w = min(x2,1024) - x
            h = min(y2,1024) - y
            bbs_list.append({"x" : x, "y" : y, "w" : w, "h" : h, "label" : label, "uid" : uid})
    return pd.DataFrame(bbs_list)

def postprocess(split_raw_json_path, definitive_folder, label_map_json, save_path):
    """Implements the postprocessing pipeline"""
    log = LoggerSingleton()
    log.name="Postprocessing"
    predicted_patches_folder_path = join(definitive_folder, "test_pred_masks")
    predicted_dataset = PredictedDataset(split_raw_json_path, predicted_patches_folder_path)
    metrics = []
    for dis_id, tile_id, tile_dict, pred_mask in predicted_dataset:
        gt_bbs_df = get_bbs_form_json(tile_dict["dmg_json"])
        table = pd.DataFrame(gt_bbs_df.value_counts(subset=["label"]))
        manager = MetricManager(bld_labels=[1],dmg_labels=[0,1,2,3,4])
        curr_metrics = manager.compute_pred_metrics(gt_bbs_df, tile_dict["bld_mask"],
                                                    tile_dict["dmg_mask"], pred_mask)
        metrics.append(curr_metrics)
        save_results(gt_bbs_df, dis_id, tile_id, tile_dict, table, label_map_json, save_path)
    MetricManager.save_metrics(metrics, save_path, file_prefix="predicted")