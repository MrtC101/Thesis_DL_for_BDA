"""Train Pipeline

This module establishes the sequence of execution of
scripts inside `preprocessing`, `train` and `test`
packages, to implement the data preprocessing,
model training and testing pipeline for this project.
"""
import os
import sys
from os.path import join
from utils.common.logger import LoggerSingleton
from preprocessing.prepare_folder.clean_folder import delete_not_in
from preprocessing.prepare_folder.create_label_masks import create_masks
from preprocessing.prepare_folder.delete_extra import leave_only_n
from preprocessing.raw.data_stdv_mean import create_data_dicts
from preprocessing.raw.split_raw_dataset import split_dataset
from preprocessing.sliced.make_smaller_tiles import slice_dataset
from preprocessing.sliced.split_sliced_dataset import split_sliced_dataset
from preprocessing.shards.split_shard_dataset import split_shard_dataset
from preprocessing.shards.make_data_shards import create_shards
from train.training import train_model

os.environ["PROJ_PATH"] = "/home/mrtc101/Desktop/tesina/repo/my_siames"
os.environ["SRC_PATH"] = join(os.environ["PROJ_PATH"], "src")
os.environ["DATA_PATH"] = join(os.environ["PROJ_PATH"], "data")
os.environ["OUT_PATH"] = join(os.environ["PROJ_PATH"], "out")

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

def preprocess():
    """Pipeline sequence for data preprocessing."""
    xbd_path = join(os.environ["DATA_PATH"], "xBD")
    raw_path = join(xbd_path, "raw")
    # folder cleaning
    delete_not_in(raw_path)
    create_masks(raw_path, 1)
    leave_only_n(raw_path, 40)
    # Raw data
    split_json_path = split_dataset(raw_path, xbd_path)
    data_dicts_path = create_data_dicts(split_json_path, xbd_path)
    # Sliced
    sliced_path = join(xbd_path, "sliced")
    slice_dataset(split_json_path, sliced_path)
    split_sliced_json_path = split_sliced_dataset(
        sliced_path, split_json_path, xbd_path)
    # Sharded
    mean_stddev_json = join(
        data_dicts_path, "all_tiles_mean_stdev.json")
    shards_path = join(xbd_path, "shards")
    create_shards(split_sliced_json_path, mean_stddev_json, shards_path, 4)
    split_shard_json_path = split_shard_dataset(shards_path, xbd_path)
    return split_shard_json_path

def train(split_shard_json_path):
    """Pipeline sequence for training the model."""
    train_config = {
        'labels_dmg': [0, 1, 2, 3, 4],
        'labels_bld': [0, 1],
        'weights_seg': [1, 15],
        'weights_damage': [1, 35, 70, 150, 120],
        'weights_loss': [0, 0, 1],
        'mode': 'dmg',
        'init_learning_rate': 0.0005,  # dmg: 0.005, #UNet: 0.01,
        'device': 'cpu',
        'epochs': 2,  # 1500,
        'batch_size': 32,
        'num_chips_to_viz': 1
    }
    path_config = {
        'exp_name': 'train_UNet',  # train_dmg
        'out_dir': os.environ["OUT_PATH"],
        'shard_splits_json': split_shard_json_path,
        'label_map_json': join(os.environ["DATA_PATH"], "constants",
                               "xBD_label_map.json"),
        'starting_checkpoint_path': None
    }
    train_model(train_config, path_config)

# evaluate with test
# test_model()
# inference


if __name__ == "__main__":
    # FIRST AND UNIQUE LOGGER FROM ALL TRAINING PIPELINE
    log = LoggerSingleton("Training Pipeline",
                          join(os.environ["OUT_PATH"], "console_logs"))
    split_shard_json_path = preprocess()
    #split_shard_json_path = join(os.environ["DATA_PATH"], "xBD", "shards")
    train(split_shard_json_path)
