import os
import sys
import argparse
from tqdm import tqdm
from utils.common.files import is_dir
from os.path import join
from utils.common.logger import LoggerSingleton

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
log = LoggerSingleton()

"""
DISASTERS_OF_INTEREST = ('guatemala-volcano_', 'hurricane-florence_',
    'hurricane-harvey_', 'mexico-earthquake_', 'midwest-flooding_',
    'palu-tsunami_', 'santa-rosa-wildfire_', 'socal-fire_',
    'lower-puna-volcano_', 'nepal-flooding_', 'pinery-bushfire_',
    'portugal-wildfire_', 'sunda-tsunami_', 'woolsey-fire_')
"""
"""
DISASTERS_OF_INTEREST = ('midwest-flooding_','guatemala-volcano_',
    'hurricane-matthew_','hurricane-michael_', 'hurricane-florence_',
    'hurricane-harvey_', 'santa-rosa-wildfire_', 'socal-fire_',
    'lower-puna-volcano_', 'nepal-flooding_', 'pinery-bushfire_',
    'portugal-wildfire_', 'woolsey-fire_')
"""


def delete_not_in(data_path: str) -> None:
    """Deletes all files from data_path directory that don't starts with a
    disaster of interest.

    Args:
        data_path (str): Path to the xBD dataset directory 
        with subset directories.

    Raises:
        AssertionException: If Path is not a Folder

    Example:
        >>> delete_not_in("data/xBD/raw")
    """

    log.name = "Clean Folder"
    DISASTERS_OF_INTEREST = ('mexico-earthquake_', 'palu-tsunami_', 'sunda-tsunami_')
    is_dir(data_path)
    for subset in tqdm(os.listdir(data_path)):
        subset_path = join(data_path, subset)
        log.info(f"Cleaning {subset_path}/ folder.")
        for folder in os.listdir(subset_path):
            folder_path = join(subset_path, folder)
            if(not os.path.isfile(folder_path)):
                for file in os.listdir(folder_path):
                    if not file.startswith(DISASTERS_OF_INTEREST):
                        os.remove(join(folder_path, file))
            else:
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Deletes all files from data_path directory that don\'t \
            starts with a disaster of interest.')
    parser.add_argument(
        'data_path',
        help=('Path to the directory that contains both the `images` and \
              `labels` folders.')
    )
    args = parser.parse_args()
    delete_not_in(args.data_path)
