# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

import random
import argparse
from tqdm import tqdm
from itertools import chain
from utils.pathManagers.rawManager import RawPathManager
from utils.loggers.console_logger import LoggerSingleton, TqdmToLog
log = LoggerSingleton()


def leave_only_n(data_path: dict, n: int) -> None:
    """Pick n random disaster tiles and deletes all the rest of files from
    disaster Dataset.

    Args:
        data_path: Path to the directory to the xBD dataset that contains
        subsets of xBD dataset.
        n: number of disasters that will be left in the folder.

    Example:
        >>> leave_only_n("data/xBD/raw",45)
    """

    log.name = "Delete Extra"
    data = RawPathManager.load_paths(data_path)

    tot_tiles = [tile for tiles in data.values() for tile in tiles.values()]
    total_tiles = len(tot_tiles)
    if (total_tiles > n):
        log.info(f"Deleting {total_tiles-n} tiles of {total_tiles} total tiles.")

        ids = [random.randint(0, total_tiles) for _ in range(n)]
        for i in tqdm(range(total_tiles), file=TqdmToLog(log)):
            if (i not in ids):
                files_to_remove = [file for time in tot_tiles[i].values() for file in time.values()]
                for file in files_to_remove:
                    os.remove(file)
        log.info(f"Files {total_tiles-n}  removed. {n} tiles left.")
    else:
        log.info(f"There are {total_tiles} total tiles. From {int(total_tiles/2)} disasters.\
                  Skipped..")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pick n random disaster's tiles and deletes all the rest \
            of files from disaster Dataset.")
    parser.add_argument(
        'data_path',
        help=("Path to xBD dataset folder that contains subsets. \
              Ex: data/xBD/raw")
    )
    parser.add_argument(
        '-n', '--only_n',
        type=int,
        help=('Only leaves n disaster tiles.')
    )
    args = parser.parse_args()
    leave_only_n(args.data_path, args.n)
