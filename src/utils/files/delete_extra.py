import os
import sys
if(os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.logger import get_logger
l = get_logger("delete_extra")

import random
import argparse
from tqdm import tqdm
from itertools import chain
from utils.files.path_manager import RawPathManager

def leave_only_n(data_path : dict, n : int) -> None:
    """
    Pick n random tiles and deletes all the rest of files from disaster Dataset.
    """
    data = RawPathManager.load_dataset(data_path)

    tiles = list(chain.from_iterable(data))
    total_tiles = len(tiles)
    
    l.info(f"Deleting {n} tiles of {total_tiles} total tiles.")

    ids = [random.randint(0, total_tiles) for _ in range(n)]    

    files_to_remove = list(chain.from_iterable(chain.from_iterable(tiles)))
    l.info(f"{len(files_to_remove)} files to remove.")
    for i in tqdm(range(total_tiles)):
        if(i not in ids):
            for file in files_to_remove:
                os.remove(file)
    l.info("Files removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pick n random tiles and deletes all the rest of files from disaster Dataset.')
    parser.add_argument(
        'data_path',
        help=('Path to xBD dataset folder.')
    )
    parser.add_argument(
        '-n', '--only_n',
        type=int,
        help=('Only leaves n tiles.')
    )
    args = parser.parse_args()
    leave_only_n(args.data_path,args.n)