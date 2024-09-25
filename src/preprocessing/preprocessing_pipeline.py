# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
from preprocessing.data_augmentation.cutmix import do_cutmix
from utils.common.pathManager import FilePath
from utils.loggers.console_logger import LoggerSingleton
from preprocessing.raw.gready_samp import greedy_split_dataset
from preprocessing.prepare_folder.create_label_masks import create_masks
from preprocessing.raw.split_raw_dataset import stratified_split_dataset
from preprocessing.data_augmentation.augmentation import make_augmentations
from preprocessing.raw.data_stdv_mean import compute_pixel_weights, create_data_dicts
from preprocessing.sliced.make_smaller_tiles import slice_dataset
from preprocessing.sliced.split_sliced_dataset import split_sliced_dataset


def log_Title(title: str):
    """Prints a Title throw logger"""
    log = LoggerSingleton()
    log.info("="*50)
    log.info(f"{title.upper()}...")
    log.info("="*50)


def preprocess() -> dict:
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
    pre_path = FilePath(os.environ["OUT_PATH"]).join("preprocessing")
    LoggerSingleton("PREPROCESSING", folder_path=pre_path)

    # LOAD PARAMETERS
    params = FilePath(os.environ["EXP_PATH"]) \
        .join("params.yml").read_yaml()

    log_Title("Creating target masks")
    xbd_path = FilePath(os.environ["XBD_PATH"])
    create_masks(xbd_path)

    data_path = FilePath(os.environ["DATA_PATH"])
    if data_path.is_dir():
        data_path.remove()
    split_prop = {"train": 0.9, "test": 0.1}
    total_tiles = params["preprocessing"]["img_num"]
    # Estrategía de gready sampling

    if params["preprocessing"]["custom_sampling"]:
        log_Title("Split disasters with greedy approach")
        tile_splits_json_path = greedy_split_dataset(xbd_path, data_path, total_tiles, split_prop)
        compute_pixel_weights(data_path, tile_splits_json_path)
    else:
        # Raw data
        log_Title("Split disasters")
        data_path.create_folder()
        disasters_of_interest = tuple(params["preprocessing"]["disasters"])
        tile_splits_json_path = stratified_split_dataset(xbd_path,
                                                         data_path,
                                                         split_prop,
                                                         disasters_of_interest,
                                                         total_tiles)

    # Data augmentation
    if (params["preprocessing"]["cutmix"]):
        log_Title("Balance dataset with cutmix")
        out_path = FilePath(os.environ["OUT_PATH"])
        tile_splits_json_path = do_cutmix(tile_splits_json_path, data_path, out_path)

    if (int(params["preprocessing"]["aug_num"]) > 0):
        log_Title("Data augmentation")
        aug_num = params["preprocessing"]["aug_num"]
        tile_splits_json_path = make_augmentations(tile_splits_json_path,
                                                   data_path, aug_num)

    log_Title("Creating data statistics")
    data_dicts_path = create_data_dicts(tile_splits_json_path,
                                        data_path)
    mean_std_json_path = data_dicts_path.join("all_tiles_mean_stddev.json")

    # Cropping
    log_Title("Creating data patches")
    patch_path = data_path.join("sliced")
    slice_dataset(tile_splits_json_path, patch_path)

    log_Title("Split patches")
    patch_split_json_path = split_sliced_dataset(patch_path,
                                                 tile_splits_json_path,
                                                 data_path,
                                                 "sliced_splits.json")
    paths = {
        "tile_splits_json_path": tile_splits_json_path,
        "patch_split_json_path": patch_split_json_path,
        "mean_std_json_path": mean_std_json_path
    }
    return paths
