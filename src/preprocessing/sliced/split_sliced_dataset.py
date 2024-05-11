import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.logger import get_logger
l = get_logger("Split_sliced")

import argparse
from utils.common.files import dump_json
from utils.pathManagers.slicedManager import SlicedPathManager

def split_sliced_dataset(sliced_path : str ,raw_split_json : str, out_path : str) -> None:
    """
        Splits the xbd dataset in train and test sets. (80%,10%,10%)
    """
    # creates splits folder    
    split_path = os.path.join(out_path,"splits")
    os.makedirs(split_path,exist_ok=True)

    # loads sliced dataset
    sliced_dict = SlicedPathManager().load_paths(sliced_path,raw_split_json)
    
    split_file = os.path.join(split_path,"sliced_splits.json")
    dump_json(split_file,sliced_dict)
    return split_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Creates a json files that shows the splits for patches..')
    parser.add_argument(
        'sliced_path',
        help=('Path to the sliced dataset.')
    )
    parser.add_argument(
        'raw_split_json',
        help=('Path to the raw split json file')
    )
    parser.add_argument(
        'out_path',
        help=('Path to he folder to create the json file output.')
    )
    args = parser.parse_args()
    split_sliced_dataset(args.sliced_path, args.raw_split_json, args.out_path)