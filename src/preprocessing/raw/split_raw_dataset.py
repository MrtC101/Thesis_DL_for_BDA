# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
    
import argparse
import math
from tqdm import tqdm
from random import shuffle
from collections import defaultdict
from utils.common.files import dump_json
from utils.pathManagers.rawManager import RawPathManager, DisasterDict
from utils.loggers.console_logger import LoggerSingleton
log = LoggerSingleton()

def split_dataset(raw_path: str, out_path: str,sets :dict) -> str:
    """Creates a dictionary representing the splits of the xBD dataset.
        The dictionary is saved in a new json file\
        `raw_splits.json` inside `out_path/splits` folder.
        The splits are made based on `sets` dictionary.

    Args:
        raw_path: Path to the folder containing subsets from the xBD dataset.
        out_path: Path to the folder where the new JSON file will be saved.
        sets: Dictionary of pairs (key,value) where the key is the split set's name and 
        it's value is the proportion of the dataset. 
    
    Return:
        str: Path to the new `raw_splits.json` file.

    Example:
        >>> split_dataset("data/xBD/raw","data/xBD")
    """
    log.name = "Split Raw Dataset"
    # creates splits folder
    split_path = os.path.join(out_path, "splits")
    os.makedirs(split_path, exist_ok=True)

    # loads raw dataset
    xbd_raw: DisasterDict = RawPathManager.load_paths(raw_path)
    tiles_file = os.path.join(split_path, "all_disaster.json")
    dump_json(tiles_file, xbd_raw)
 
    total = sum([val for val in sets.values()])
    assert total == 1.0, f"Sets proportions must sum 1.0, the current total is {total}"

    splits_dict = defaultdict(lambda : defaultdict(lambda : {}))
    for disaster_name, tiles_dict in tqdm(xbd_raw.items()):

        tiles_ids = list(tiles_dict.keys())
        shuffle(tiles_ids)

        if(len(tiles_ids) < len(sets)):
            raise Exception(f"{disaster_name} disaster number must be more than {len(sets)}.")
        
        # works fine if len(tiles_ids) >= 10 for each disaster
        ids = {}
        last_top = 0
        for i, (name, proportion) in enumerate(sets.items()):
            current_top = last_top + math.floor(proportion * len(tiles_ids))
            if(i < len(sets)-1):
                ids[name] = sorted(tiles_ids[last_top:current_top])
            else:
                ids[name] = sorted(tiles_ids[last_top:])
            last_top = current_top
    
        msg = f"{disaster_name} length {len(tiles_ids)}, "
        for set in sets.keys():
            for id in ids[set]:
                splits_dict[set][disaster_name][id] = tiles_dict[id]
            msg+=f"{set} {len(ids[set])} " #For loggin strings
        
        log.info(msg)

    split_file = os.path.join(split_path, "raw_splits.json")
    dump_json(split_file, splits_dict)
    return split_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create masks for each label json file for disasters\
              specified at the top of the script.')
    parser.add_argument(
        'raw_path',
        help=('Path to the directory that contains all files related with\
               xBD dataset.')
    )
    parser.add_argument(
        'out_path',
        help=('Path for the output json file.')
    )
    args = parser.parse_args()
    split_dataset(args.out_path)
