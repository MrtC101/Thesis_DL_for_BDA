# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import sys
if(os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.logger import get_logger
l = get_logger("Slice_Dataset")

import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader
from utils.files.common import read_json
from utils.files.datasets import SliceDataset

def load_split(set_name,splits_json,output_path):

    splits_all_disasters = read_json(splits_json)
    
    # Tiles num count
    split = splits_all_disasters[set_name]
    set_length = sum(len(tile) for tile in split.values())
    
    l.info(f'{set_name} dataset length before cropping: {set_length}.')

    dataset = SliceDataset(set_name,split,output_path)

    l.info(f'{set_name} dataset length after cropping: {len(dataset)}.')
    return dataset

def slice_dataset(splits_json, output_path, batch_size):
    
    def iterate_and_slice(split_name):
        l.info(f'Starting slicing for {split_name}')      
        dataset = load_split(split_name,splits_json, output_path)
        dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=8)
        
        for _ in tqdm(dataloader):
            continue

        l.info(f'Done slicing for {split_name}')
    
    #iterate_and_slice("train")
    with ThreadPoolExecutor(max_workers=2) as executor:
        for split_name in ["train","val"]:
            executor.submit(iterate_and_slice, split_name)

    l.info(f'Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create slices from a xBD dataset.')
    parser.add_argument(
        'split_json_path',
        type=str,
        help=('Path to the json file with the train/val/test split.')
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help=('Path to folder for new sliced data.')
    )
    parser.add_argument(
        '-b','--batch_size',
        type = int,
        default = 1,
        help=('Size of the batch of images for augmentation.')
    )
    args = parser.parse_args()
    slice_dataset(args.split_json_path, args.output_dir, args.batch_size)