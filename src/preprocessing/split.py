import os
import sys
if(os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.logger import get_logger
l = get_logger("delete_extra")

import json
import os
from random import shuffle
import math

import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter, defaultdict
import numpy as np
import argparse

parser = argparse.ArgumentParser(
        description='Create masks for each label json file for disasters specified at the top of the script.')
parser.add_argument(
    'xbd_root',
    help=('Path to the directory that contains both the `images` and `labels` folders. '
            'The `targets_border{border_width}` folder will be created if it does not already exist.')
)
parser.add_argument(
    'label_dir',
    help=('Path to the directory that contains both the `images` and `labels` folders. '
            'The `targets_border{border_width}` folder will be created if it does not already exist.')
)

args = parser.parse_args()
earthquake_disasters = ['mexico-earthquake', 'palu-tsunami', 'sunda-tsunami']

xbd_root = args.xbd_root

subset_disasters = earthquake_disasters
label_dirs = [
    args.label_dir
]
all_files = defaultdict(list)  # subset disaster to list of files (no extension)

subset_disasters_tup = tuple(subset_disasters)

for label_dir in label_dirs:
    for p in tqdm(os.listdir(label_dir)):

        if not p.startswith(subset_disasters_tup):
            continue
        
        if not p.endswith('_post_disaster.json'):
            continue
        
        full_path = os.path.join(label_dir, p)
        rel_path = full_path.split(xbd_root)[1]
        
        # example: hurricane-matthew_00000000_post_disaster.json
        disaster_name = p.split('_')[0]
        file = rel_path.split('_post_disaster.json')[0]
        
        all_files[disaster_name].append(file.strip("/"))

for disaster_name, files in all_files.items():
    print(f'{disaster_name}, {len(files)}')

other_splits = defaultdict(dict)

for disaster_name, files in all_files.items():

    shuffle(files)
    
    num_train_tiles = math.ceil(0.8 * len(files))
    num_val_tiles = math.ceil(0.1 * len(files))
    
    other_splits[disaster_name]['train'] = sorted(files[:num_train_tiles])
    other_splits[disaster_name]['val'] = sorted(files[num_train_tiles:num_train_tiles + num_val_tiles])
    other_splits[disaster_name]['test'] = sorted(files[num_train_tiles + num_val_tiles:])

    print(f"{disaster_name}, train {len(other_splits[disaster_name]['train'])}, val {len(other_splits[disaster_name]['val'])}, test {len(other_splits[disaster_name]['test'])}")

if(not os.path.exists('./constants/splits/')):
    os.mkdir('./constants/splits/')

with open('./constants/splits/final_mdl_all_disaster_splits.json', 'w') as f:
    json.dump(other_splits, f, indent=4)