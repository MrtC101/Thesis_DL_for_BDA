# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import math
from tqdm import tqdm
from random import shuffle
from collections import defaultdict
from utils.common.pathManager import FilePath
from utils.pathManagers.rawManager import RawPathManager
from utils.loggers.console_logger import LoggerSingleton
log = LoggerSingleton()


def split_dataset(raw_path: FilePath, out_path: FilePath,
                  splits: dict) -> FilePath:
    """Creates a dictionary representing the splits of the xBD dataset.
        The dictionary is saved in a new json file\
        `raw_splits.json` inside `out_path/splits` folder.
        The splits are made based on `splits` dictionary.

    Args:
        raw_path: Path to the folder containing subsets from the xBD dataset.
        out_path: Path to the folder where the new JSON file will be saved.
        splits: Dictionary of pairs (key,value) where the key is the split
        set's name and it's value is the proportion of the dataset.

    Return:
        FilePath: Path to the new `raw_splits.json` file.

    Example:
        >>> split_dataset("data/xBD/raw","data/xBD")
    """

    # creates splits folder
    split_path = out_path.join("splits")
    split_path.create_folder()

    # loads raw dataset
    data_dict = RawPathManager.load_paths(raw_path)
    tiles_file = split_path.join("all_disaster.json")
    tiles_file.save_json(data_dict)

    assert sum([val for val in splits.values()]) == 1.0, \
        "splits proportions must sum 1.0"

    splits_dict = defaultdict(lambda: defaultdict(lambda: {}))
    for disaster_name, tiles_dict in tqdm(data_dict.items()):

        tiles_ids = list(tiles_dict.keys())
        shuffle(tiles_ids)

        if (len(tiles_ids) < len(splits)):
            raise Exception(f"{disaster_name} disaster number must be more\
                             than {len(splits)}.")

        # works fine if len(tiles_ids) >= 10 for each disaster
        ids = {}
        last_top = 0
        for i, (name, proportion) in enumerate(splits.items()):
            current_top = last_top + math.floor(proportion * len(tiles_ids))
            if i < len(splits) - 1:
                ids[name] = sorted(tiles_ids[last_top:current_top])
            else:
                ids[name] = sorted(tiles_ids[last_top:])
            last_top = current_top

        msg = f"{disaster_name} length {len(tiles_ids)}, "
        for set in splits.keys():
            for id in ids[set]:
                splits_dict[set][disaster_name][id] = tiles_dict[id]
            msg += f"{set} {len(ids[set])} "  # For loggin strings

        log.info(msg)

    split_file = split_path.join("raw_splits.json")
    split_file.save_json(splits_dict)
    return split_file
