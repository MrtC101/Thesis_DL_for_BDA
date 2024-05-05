import os
import sys
if(os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.logger import get_logger
l = get_logger("delete_extra")


import argparse
from os.path import join
from tqdm import tqdm

#DISASTERS_OF_INTEREST = ('guatemala-volcano_', 'hurricane-florence_', 'hurricane-harvey_', 'mexico-earthquake_', 'midwest-flooding_', 'palu-tsunami_', 'santa-rosa-wildfire_', 'socal-fire_', 'lower-puna-volcano_', 'nepal-flooding_', 'pinery-bushfire_', 'portugal-wildfire_', 'sunda-tsunami_', 'woolsey-fire_')
#DISASTERS_OF_INTEREST = ('midwest-flooding_','guatemala-volcano_', 'hurricane-matthew_','hurricane-michael_', 'hurricane-florence_', 'hurricane-harvey_', 'santa-rosa-wildfire_', 'socal-fire_', 'lower-puna-volcano_', 'nepal-flooding_', 'pinery-bushfire_', 'portugal-wildfire_', 'woolsey-fire_')
DISASTERS_OF_INTEREST = ('mexico-earthquake_', 'palu-tsunami_', 'sunda-tsunami_')

def delete_not_in(data_path : str) -> None:
    """
        Deletes all files from data_path directory that are not of interest.
    """
    assert os.path.isdir(data_path), f"{data_path} is not a directory."
    
    for subset in tqdm(os.listdir(data_path)):
        l.info(f"Cleaning {subset}/ folder.")
        for folder in os.listdir(subset):
            folder_path = join(data_path,folder)
            for file in os.listdir(folder_path):
                if not file.startswith(DISASTERS_OF_INTEREST):
                    os.remove(join(folder_path,file))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Delete all files that are not related with disasters of interest.')
    parser.add_argument(
        'data_path',
        help=('Path to the directory that contains both the `images` and `labels` folders.')
    )
    args = parser.parse_args()
    delete_not_in(args.data_path)

        