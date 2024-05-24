# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys
import argparse
import math
from tqdm import tqdm
from random import shuffle
from utils.common.files import dump_json
from utils.pathManagers.rawManager import RawPathManager, DisasterDict
from utils.common.logger import LoggerSingleton

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
log = LoggerSingleton()


def split_dataset(raw_path: str, out_path: str) -> str:
    """Creates a dictionary representing the splits (train 80%,val 10%,\
      test 10%) of the xBD dataset. The dictionary is saved in a new json file\
      `raw_splits.json` inside `out_path/splits` folder.

    Args :
        raw_path: Path to the folder containing subsets from the xBD dataset.
        out_path: Path to the folder where the new JSON file will be saved.

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

    splits_dict = {
        "train": {},
        "val": {},
        "test": {}
    }
    for disaster_name, tiles_dict in tqdm(xbd_raw.items()):

        tiles_ids = list(tiles_dict.keys())
        shuffle(tiles_ids)

        if(len(tiles_ids) < 3):
            raise Exception(f"{disaster_name} disasters are less than 3.")
        elif(len(tiles_ids) < 10):
            train_top = len(tiles_ids) -2
            val_top = len(tiles_ids) -1
        else:
            # works fine if len(tiles_ids) >= 10 for each disaster
            train_top = math.ceil(0.8 * len(tiles_ids))
            val_top = train_top + math.floor(0.1 * len(tiles_ids))

        ids = {
            "train": sorted(tiles_ids[0:train_top]),
            "val": sorted(tiles_ids[train_top:val_top]),
            "test": sorted(tiles_ids[val_top:])
        }

        for set in ["train", "val", "test"]:
            splits_dict[set][disaster_name] = {}
            for id in ids[set]:
                splits_dict[set][disaster_name][id] = tiles_dict[id]

        log.info(f"{disaster_name} length {len(tiles_ids)},\
                 train {len(ids['train'])},\
                 val {len(ids['val'])},\
                 test {len(ids['test'])}")

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
