# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
from typing import List, Tuple
from os.path import join
from utils.common.pathManager import FilePath
from utils.loggers.console_logger import LoggerSingleton
from preprocessing.prepare_folder.clean_folder import delete_not_in
from preprocessing.prepare_folder.create_label_masks import create_masks
from preprocessing.raw.split_raw_dataset import stratified_split_dataset
from preprocessing.data_augmentation.augmentation import make_augmentations
from preprocessing.raw.data_stdv_mean import create_data_dicts
from preprocessing.sliced.make_smaller_tiles import slice_dataset
from preprocessing.sliced.split_sliced_dataset import split_sliced_dataset


def log_Title(title: str):
    """Prints a Title throw logger"""
    log = LoggerSingleton()
    log.info("="*50)
    log.info(f"{title.upper()}...")
    log.info("="*50)


def preprocess(total_tiles: int, num_aug: int,
               disasters_of_interest: List[str]) -> Tuple[str, str, str, str, str]:
    """
    Pipeline sequence for data preprocessing.

    Args:
        total_tiles (int): Total number of tiles to use for training the model.
        num_aug (int): Number of augmented images to create.
        disasters_of_interest (List[str]): List of disaster identifiers 
            as strings for each disaster type that will be used for training 
            the model. For example: ['midwest-flooding'].

    Returns:
        Tuple[str, str, str, str, str]: Paths to the JSON files representing 
        splits of tiles, splits of patches, and statistical data from tiles.
    """
    pre_path = FilePath(join(os.environ["OUT_PATH"])).join("preprocessing")
    LoggerSingleton("PREPROCESSING", folder_path=pre_path)

    # folder cleaning
    # log_Title("Deleting disasters that are not of interest")
    # delete_not_in(raw_path, disasters_of_interest)

    log_Title("Creating target masks")
    xbd_path = FilePath(join(os.environ["XBD_PATH"]))
    create_masks(xbd_path)

    # Raw data
    log_Title("Split disasters")
    data_path = FilePath(join(os.environ["DATA_PATH"]))
    split_prop = {"train": 0.9, "test": 0.1}
    tile_splits_json_path = stratified_split_dataset(xbd_path, data_path,
                                                     split_prop, total_tiles)

    # Data augmentation
    log_Title("Data augmentation")
    aug_tile_split_json_path = make_augmentations(tile_splits_json_path,
                                                  xbd_path, num_aug)

    log_Title("Creating data statistics")
    data_dicts_path = create_data_dicts(aug_tile_split_json_path, xbd_path)
    mean_std_json_path = data_dicts_path.join("all_tiles_mean_stddev.json")

    # Cropping
    log_Title("Creating data patches")
    patch_path = xbd_path.join("sliced")
    slice_dataset(aug_tile_split_json_path, patch_path)

    log_Title("Split patches")
    patch_split_json_path = split_sliced_dataset(patch_path,
                                                 tile_splits_json_path,
                                                 xbd_path,
                                                 "sliced_splits.json")

    log_Title("Split augmented patches")
    aug_patch_split_json_path = split_sliced_dataset(patch_path,
                                                     aug_tile_split_json_path,
                                                     xbd_path,
                                                     "aug_sliced_splits.json")
    
    return (tile_splits_json_path, patch_split_json_path,
            aug_tile_split_json_path, aug_patch_split_json_path,
            mean_std_json_path)
