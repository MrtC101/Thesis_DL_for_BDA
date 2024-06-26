# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
"""Train Pipeline

This module establishes the sequence of execution of
scripts inside `preprocessing`, `train` and `test`
packages, to implement the data preprocessing,
model training and testing pipeline for this project.
"""
import os
import sys
from os.path import join

# Environment variables
os.environ["PROJ_PATH"] = "/home/mrtc101/Desktop/tesina/repo/base_siames"
os.environ["SRC_PATH"] = join(os.environ["PROJ_PATH"], "src")
os.environ["DATA_PATH"] = join(os.environ["PROJ_PATH"], "data")
os.environ["OUT_PATH"] = join(os.environ["PROJ_PATH"], "out")

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.logger import LoggerSingleton
from preprocessing.prepare_folder.clean_folder import delete_not_in
from preprocessing.prepare_folder.create_label_masks import create_masks
from preprocessing.prepare_folder.delete_extra import leave_only_n
from preprocessing.raw.data_stdv_mean import create_data_dicts
from preprocessing.raw.split_raw_dataset import split_dataset
from preprocessing.sliced.make_smaller_tiles import slice_dataset
from preprocessing.sliced.split_sliced_dataset import split_sliced_dataset
from train.training import train_model
from train.validation import test_model

# Configuration dictionary for pre processing step
pre_config = {
    "disaster_num": 1000,
    "border_with": 1,
}

# Configuration dictionary for model training
model_config = {
    'labels_dmg': [0, 1, 2, 3, 4],
    'labels_bld': [1],
    'weights_seg': [1, 15],
    'weights_damage': [1, 35, 70, 150, 120],
    'weights_loss': [0, 0, 1],
    'mode': 'dmg',
    'init_learning_rate': 0.0005,  # dmg: 0.005, #UNet: 0.01,
    'device': 'cpu',
    'epochs': 1,  # 1500,
    'batch_size': 1,
    'num_chips_to_viz': 1,
    'torch_threads': 1,
    'torch_op_threads': 1,
    'batch_workers': 1,
}

# Configuration dictionary for paths used during model training
path_config = {
    'exp_name': 'train_UNet',  # train_dmg
    'out_dir': os.environ["OUT_PATH"],
    "dataset_path": '',
    'label_map_json': join(os.environ["DATA_PATH"], "constants","xBD_label_map.json"),
    'starting_checkpoint_path': None
}


def log_Title(title: str):
    """Prints a Title throw logger"""
    log.info("="*50)
    log.info(f"{title.upper()}...")
    log.info("="*50)


def preprocess(disaster_num, border_with):
    """Pipeline sequence for data preprocessing."""
    
    # folder cleaning
    log_Title("deleting disasters that are not of interest")
    xbd_path = join(os.environ["DATA_PATH"], "xBD")
    raw_path = join(xbd_path, "raw")
    delete_not_in(raw_path)
    
    log_Title("creating target masks")
    create_masks(raw_path, border_with)
    
    log_Title("deleting extra disasters")
    leave_only_n(raw_path, disaster_num)
    
    # Raw data
    log_Title("split disasters")
    split_json_path = split_dataset(raw_path, xbd_path,{
        "train": 0.9,
        "test": 0.1
    })

    log_Title("creating data statistics")
    data_dicts_path = create_data_dicts(split_json_path, xbd_path)
    mean_std_json_path = os.path.join(data_dicts_path,"all_tiles_mean_stddev.json")
    #Imbalance treatment goes here

    #Data augmentation goes here
    
    # Cropping
    log_Title("creating data patches")
    sliced_path = join(xbd_path, "sliced")
    slice_dataset(split_json_path, sliced_path)
    
    log_Title("split patches")
    split_sliced_json_path = split_sliced_dataset(sliced_path, split_json_path, xbd_path)
    
    return split_sliced_json_path, data_dicts_path


if __name__ == "__main__":
    # FIRST declaration of the logger used throw pipeline
    log = LoggerSingleton("Training Pipeline",
                          folder_path=join(os.environ["OUT_PATH"], "console_logs"))
    
    #dataset_path,statistics_path = preprocess(**pre_config)

    dataset_path = "/home/mrtc101/Desktop/tesina/repo/base_siames/data/xBD/splits/sliced_splits.json"
    statistics_path = "/home/mrtc101/Desktop/tesina/repo/base_siames/data/xBD/dataset_statistics/all_tiles_mean_stddev.json"

    path_config["dataset_path"] = dataset_path
    path_config["statistics_path"] = statistics_path
    log_Title("training and validating model")
    train_model(model_config, path_config)

    #hiper_tuning(){
    #   dataset_path = preprocess(**pre_config)
    #   cross_validation(model_config, path_config){
    #       train_model(model_config, path_config)
    #       test_model(model_config, path_config)
    #   }
    #}