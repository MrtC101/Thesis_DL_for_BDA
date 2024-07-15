# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
from utils.common.pathManager import FilePath
from utils.pathManagers.slicedManager import SlicedPathManager
from utils.loggers.console_logger import LoggerSingleton

log = LoggerSingleton()


def split_sliced_dataset(sliced_path: FilePath, raw_split_json: FilePath,
                         out_path: FilePath) -> None:
    """Uses the raw_split JSON file to split the raw dataset to create a new
    sliced_splits.json file.
    Args:
        sliced_path: Path to the sliced data folder.
        raw_split_json: Path to the `raw_split.json` file.
        out_path: Path to the folder where the new json file will be stored.
        if the folder "splits" do not exist it will be created.
    """
    # creates splits folder
    split_path = out_path.join("splits")
    out_path.create_folder()

    # loads sliced dataset
    sliced_dict = SlicedPathManager().load_paths(sliced_path, raw_split_json)
    split_file = split_path.join("sliced_splits.json")
    split_file.save_json(sliced_dict)
    log.info(f"New split saved {split_file}")
    return split_file
