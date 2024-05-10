# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
make_data_shards.py

This is an additional pre-processing step after tile_and_mask.py to cut chips out of the tiles
and store them in large numpy arrays, so they can all be loaded in memory during training.

The train and val splits will be stored separately to distinguish them.

This is an improvement on the original approach of chipping during training using LandsatDataset, but it is an
extra step, so each new experiment requiring a different input size/set of channels would need to re-run
this step. Data augmentation is still added on-the-fly.

Example invocation:
```
export AZUREML_DATAREFERENCE_wcsorinoquia=/boto_disk_0/wcs_data/tiles/full_sr_median_2013_2014

python data/make_chip_shards.py --config_module_path training_wcs/experiments/elevation/elevation_2_config.py --out_dir /boto_disk_0/wcs_data/shards/full_sr_median_2013_2014_elevation
```
"""
import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.logger import get_logger
l = get_logger("Compute data from images")

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import argparse
from tqdm import tqdm
import os
import sys
import math
import numpy as np
from torch.utils.data import DataLoader
from utils.common.files import read_json
from utils.datasets.inference_datasets import ShardDataset

def create_shard(dataloader, num_shards):
    """Iterate through the dataset to produce shards of chips as numpy arrays, for imagery input and labels.

    Args:
        dataset: an instance of LandsatDataset, which when iterated, each item contains fields
                    'chip' and 'chip_label'
        data = {'pre_image': pre_img, 'post_image': post_img, 'building_mask': mask, 'damage_mask': damage_class}

        num_shards: number of numpy arrays to store all chips in

    Returns:
        returns a 2-tuple, where
        - the first item is a list of numpy arrays of dimension (num_chips, channel, height, width) with
          dtype float for the input imagery chips
        - the second item is a list of numpy arrays of dimension (num_chips, height, width) with
          dtype int for the label chips.
    """
    image_patches = defaultdict([])
    for data in tqdm(dataloader):
        for key,img in data.items():
            image_patches[key].append(img)
        # not using chip_id and chip_for_display fields

    num_chips = len(image_patches.values()[0])
    l.info(f'Created {num_chips} chips.')

    patches_per_shards = math.ceil(num_chips / num_shards)
    shard_idx = []
    for i in range(num_shards):
        shard_idx.append(
            (i * patches_per_shards, (1 + i) * patches_per_shards)
        )

    l.info('Stacking imagery and label chips into shards')
    shard_list = defaultdict()
    for begin_idx, end_idx in shard_idx:
        if begin_idx < num_chips:
            for key,patches in image_patches.items():
                shard = patches[begin_idx:end_idx]
                shard = np.stack(shard, axis=0)
                l.info(f'dim of {key} input patch shard is {shard.shape}, dtype is {shard.dtype}')
                shard_list[f"{key}_shard"].append(shard)
    return shard_list

def save_shards(out_dir, set_name, shard_list_dict):
    os.makedirs(out_dir, exist_ok=True)  
    for file_id,shard_list in shard_list_dict.items():
        for i_shard in range(len(shard_list)):
            shard_path = os.path.join(out_dir, f'{set_name}_{file_id}_{i_shard}.npy')
            np.save(shard_path, shard_list[i_shard])
            l.info(f'Saved {shard_path}')

def load_dataset(set_name,sliced_splits_json,data_mean_stddev):
    splits = read_json(sliced_splits_json)
    data_mean_stddev = read_json(data_mean_stddev)

    split = splits[set_name]
    set_length = sum(len(tile) for tile in split.values())
    #@TODO
    dataset = ShardDataset(config['data_dir'], set_length, data_mean_stddev, transform=False, normalize=True)

    l.info(f'xBD_disaster_dataset {set_name} length: {len(dataset)}')

    return dataset

def create_shards(sliced_splits_json,out_path,batch_size,num_shards):

    def iterate_and_shard(split_name):
        dataset = load_dataset(split_name,sliced_splits_json, out_path)
        dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=8)

        l.info('Iterating through the training set to generate chips...')
        shard_dict = create_shard(dataloader, num_shards)
        save_shards(out_path, split_name,shard_dict)

        for k in shard_dict.keys():
            del shard_dict[k]

        l.info(f'Done creating shards for {split_name}')
    
    iterate_and_shard("train")
    with ThreadPoolExecutor(max_workers=2) as executor:
        for split_name in ["train","val"]:
            executor.submit(iterate_and_shard, split_name)

    l.info('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create shards from a sliced xBD dataset.')
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
    args = parser.parse_args()
    config = {'num_shards': 1,
          'out_dir': 'public_datasets/xBD/xBD_sliced_augmented_20_alldisasters_final_mdl_npy/',
          'data_dir': 'public_datasets/xBD/final_mdl_all_disaster_splits/',
          'data_splits_json': 'constants/splits/final_mdl_all_disaster_splits_sliced_img_augmented_20.json',
          'data_mean_stddev': 'constants/splits/all_disaster_mean_stddev_tiles_0_1.json'}
    create_shards(args.split_json_path, args.output_dir, args.batch_size)