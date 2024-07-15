import os
import sys
from os.path import join

from postprocessing.postprocess_pipeline import postprocess

os.environ["PROJ_PATH"] = "/home/mrtc101/Desktop/tesina/repo/hiper_siames"
os.environ["SRC_PATH"] = join(os.environ["PROJ_PATH"], "src")
os.environ["DATA_PATH"] = join(os.environ["PROJ_PATH"], "data")
os.environ["OUT_PATH"] = join(os.environ["PROJ_PATH"], "out")

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from hiper_search.hiper_parameter_search import parameter_search
from preprocessing.preprocessing_pipeline import preprocess


if __name__ == "__main__":
    # Configuration dictionary for paths used during model training
    weights_config = {
        'weights_seg': [1, 15],
        'weights_damage': [1, 35, 70, 150, 120],
        'weights_loss': [0, 0, 1],
    }
    hardware_config ={
        'device': 'cpu',
        'torch_threads': 12,
        'torch_op_threads': 12,
        'batch_workers': 0,
        'new_optimizer' : False
    }
    visual_config = {
        'num_chips_to_viz': 2,
        'labels_dmg': [ 0, 1, 2, 3, 4],
        'labels_bld': [1], # not include 0 because is binary 
    }
    pre_config = {
        "disaster_num": 20,
        "border_width": 1,
    }
    post_config = {
        
    }
    configs = dict(
        list(weights_config.items()) +
        list(hardware_config.items()) +
        list(visual_config.items()) +
        list(pre_config.items())
                   )
    split_raw_json_path, split_sliced_json_path, mean_std_json_path = preprocess(**pre_config)
    parameter_search(split_sliced_json_path, mean_std_json_path, configs)
    save_path = os.path.join(os.environ['OUT_PATH'],"postprocessing")
    pred_path = os.path.join(os.environ['OUT_PATH'],"definitive_model")
    postprocess(split_raw_json_path, pred_path, save_path)