import os
import sys
import random
import argparse
from tqdm import tqdm
from itertools import chain
from utils.pathManagers.rawManager import RawPathManager

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
from utils.common.logger import LoggerSingleton, TqdmToLog
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

    tiles = list(chain.from_iterable(data))
    total_tiles = len(tiles)
    if (total_tiles > n):
        log.info(f"Deleting {n} tiles of {total_tiles} total tiles.")

        ids = [random.randint(0, total_tiles) for _ in range(n)]

        files_to_remove = list(chain.from_iterable(chain.from_iterable(tiles)))
        log.info(f"{len(files_to_remove)} files to remove.")
        for i in tqdm(range(total_tiles), file=TqdmToLog(log)):
            if (i not in ids):
                for file in files_to_remove:
                    os.remove(file)
        log.info("Files removed.")


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
