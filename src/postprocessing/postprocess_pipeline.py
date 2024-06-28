import os
import sys
from os.path import join
import pandas as pd

from utils.datasets.predicted_dataset import PredictedDataset
from utils.pathManagers.predictedManager import PredictedPathManager
from torch.utils.data import DataLoader
# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from postprocessing.TableCount.TableCreation import analize_result
from postprocessing.plotResults import save_results
from utils.common.logger import LoggerSingleton
    
def save_metrics(metrics):
    pass

def postprocess(tile_json_path, definitive_folder,label_map_json,save_path):
    log = LoggerSingleton()
    log.name="Postprocessing"
    predicted_json_path = join(definitive_folder, "test_pred_masks")
    predicted_dataset = PredictedDataset(tile_json_path,predicted_json_path)
    metrics = []
    for dis_id, tile_id, pre_img,post_img,bld_mask,dmg_mask,pred_mask in predicted_dataset:
        curr_table = pd.DataFrame(data={
            "Damage Level":["No Damage", "Minor Damage", "Mayor Damage", "Destroyed"],
            "Buildings Number":[20,10,30,4]
            },columns=["Damage Level","Buildings Number"])
        # curr_metrics, curr_table = analize_result(pred_mask,bld_mask,dmg_mask)
        #metrics.append(curr_metrics)
        save_results(dis_id, tile_id, pre_img, post_img, pred_mask, curr_table,
                     label_map_json, save_path)
    #save_metrics(metrics)