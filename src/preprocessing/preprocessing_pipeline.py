# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.

from utils.common.pathManager import FilePath
from utils.common.timeManager import measure_time
from utils.loggers.console_logger import LoggerSingleton
from preprocessing.data_augmentation.cutmix import do_cutmix
from preprocessing.raw.gready_samp import greedy_split_dataset
from preprocessing.sliced.make_smaller_tiles import slice_dataset
from preprocessing.prepare_folder.create_label_masks import create_masks
from preprocessing.raw.split_raw_dataset import split_dataset
from preprocessing.sliced.split_sliced_dataset import split_sliced_dataset
from preprocessing.data_augmentation.augmentation import make_augmentations
from preprocessing.create_statistics.data_stdv_mean import create_data_dicts
from preprocessing.create_statistics.weigths_manager import compute_pixel_weights, create_configs


def log_Title(title: str):
    """Prints a Title throw logger"""
    log = LoggerSingleton()
    log.info("="*50)
    log.info(f"{title.upper()}...")
    log.info("="*50)


@measure_time
def preprocess(out_path: FilePath, exp_path: FilePath, xbd_path: FilePath,
               data_path: FilePath) -> dict:
    """
    Pipeline sequence for data preprocessing.

    Args:
        out_path (FilePath): Path to the output folder.
        exp_path (FilePath): Path to the experiment folder where is the params.yml file.
        xbd_path (FilePath): Path to the folder where are all xBD dataset Splits.
        data_path (FilePath): Path to the folder where image patches will be stored.

    Returns:
        dict[str]: Paths to the JSON files that represent:
        - Split train, val and test of tiles.
        - Split train, val and test of patches.
        - Tile statistics for inline normalization of patches.
    """
    # Initializes the preprocessing output file
    LoggerSingleton("PREPROCESSING", folder_path=out_path.join("preprocessing"))

    # LOAD PARAMETERS
    log_Title("Creating target masks")
    create_masks(xbd_path)

    data_path.create_folder(delete_if_exist=True)
    params: dict = exp_path.join("params.yml").read_yaml()
    total_tiles = params["preprocessing"]["img_num"]
    split_prop = {"train": 0.8, "val": 0.1, "test": 0.1}
    if params["preprocessing"]["custom_sampling"]:
        log_Title("Split disasters with greedy approach")
        tile_splits_json_path = greedy_split_dataset(xbd_path, data_path, total_tiles, split_prop)
    else:
        log_Title("Split disasters")
        disasters_of_interest = tuple(params["preprocessing"]["disasters"])
        tile_splits_json_path = split_dataset(xbd_path, data_path, split_prop,
                                              disasters_of_interest)

    # Data augmentation
    if (params["preprocessing"]["cutmix"]):
        log_Title("Balance dataset with cutmix")
        tile_splits_json_path = do_cutmix(tile_splits_json_path, data_path, out_path)

    if (int(params["preprocessing"]["aug_num"]) > 0):
        log_Title("Data augmentation")
        aug_num = params["preprocessing"]["aug_num"]
        tile_splits_json_path = make_augmentations(tile_splits_json_path, data_path, aug_num)

    log_Title("Computing pixel count and weights")
    data_dicts_path = create_data_dicts(tile_splits_json_path, data_path)
    mean_std_json_path = data_dicts_path.join("all_tiles_mean_stddev.json")

    # Cropping
    log_Title("Creating data patches")
    patch_path = data_path.join("sliced")
    slice_dataset(tile_splits_json_path, patch_path)
    patch_split_json_path = split_sliced_dataset(patch_path, tile_splits_json_path,
                                                 data_path, "sliced_splits.json")

    log_Title("Creating parameters JSON")
    params_path = exp_path.join("params.yml")
    weights = compute_pixel_weights(tile_splits_json_path)
    configs = create_configs(params_path, weights)
    out_path.join("conf_list.json").save_json(configs)

    return {
        "tile_splits_json_path": tile_splits_json_path,
        "patch_split_json_path": patch_split_json_path,
        "mean_std_json_path": mean_std_json_path
    }
