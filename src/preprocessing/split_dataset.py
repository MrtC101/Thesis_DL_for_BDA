import os
import sys
if(os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.logger import get_logger
l = get_logger("delete_extra")

import os
import argparse
import math
from tqdm import tqdm
from random import shuffle
from utils.files.common import dump_json
from utils.files.path_manager import RawPathManager, DisasterDict

from collections import Counter, defaultdict
import numpy as np


def split_dataset(xbd_path : str) -> None:
    """
        Splits the xbd dataset in train and test sets. (90%,10%)
    """
    # creates splits folder    
    split_path = os.path.join(xbd_path,"splits")
    if(not os.path.exists(split_path)):
        os.mkdir(split_path)

    # loads raw dataset
    raw_path = os.path.join(xbd_path,"raw")
    xbd_raw : DisasterDict = RawPathManager.load_dataset(raw_path)
    tiles_file = os.path.join(split_path,"all_disaster.json")
    dump_json(tiles_file,xbd_raw)

    splits_dict = {
        "train": {},
        "val": {},
        "test": {}
    }
    for disaster_name,tiles_dict in xbd_raw.items():
        
        tiles_ids = list(tiles_dict.keys())
        shuffle(tiles_ids)

        # works fine if len(tiles_ids) >= 10
        train_top = math.ceil(0.8 * len(tiles_ids))
        val_top = train_top + math.floor(0.1 * len(tiles_ids))
        
        ids = {
            "train": sorted(tiles_ids[0:train_top]),
            "val": sorted(tiles_ids[train_top:val_top]),
            "test": sorted(tiles_ids[val_top:])
        }

        for set in ["train","val","test"]:
            splits_dict[set][disaster_name] =  {}
            for id in ids[set]:
                splits_dict[set][disaster_name][id] = tiles_dict[id]
        
        l.info(f"{disaster_name} length {len(tiles_ids)}, train {len(ids['train'])}, val {len(ids['val'])}, test {len(ids['test'])}")

    split_file = os.path.join(split_path,"train_test_all_disaster_splits.json")
    dump_json(split_file,splits_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Create masks for each label json file for disasters specified at the top of the script.')
    parser.add_argument(
        'xbd_path',
        help=('Path to the directory that contains all files related with xBD dataset.')
    )
    args = parser.parse_args()
    split_dataset(args.xbd_path)