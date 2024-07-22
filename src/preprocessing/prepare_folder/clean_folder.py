# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import random
from typing import Tuple
from tqdm import tqdm
from utils.pathManagers.rawManager import RawPathManager
from utils.common.pathManager import FilePath
from utils.loggers.console_logger import LoggerSingleton

log = LoggerSingleton()


def delete_not_in(data_path: FilePath,
                  disasters_of_interest: Tuple[str]) -> None:
    """ Deletes all files from `data_path` directory that don't start with
    a disaster identifier from `disaster_of_interest` list.

    Args:
        data_path (str) : Path to the xBD dataset directory.
        disasters_of_interest (Tuple[str]) : List of disasters identifiers as
        'Mexico-earthquake'
    Raises:
        AssertionException: If Path is not a Folder
    """
    # Changes the title fo the logger.
    data_path = FilePath(data_path)
    data_path.must_be_dir()
    for subset_path in tqdm(data_path.get_folder_paths()):
        log.info(f"Cleaning {subset_path}/ folder.")
        for folder_path in subset_path.get_folder_paths():
            for file in folder_path.get_files_names():
                if not file.startswith(disasters_of_interest):
                    folder_path.join(file).remove()
    return