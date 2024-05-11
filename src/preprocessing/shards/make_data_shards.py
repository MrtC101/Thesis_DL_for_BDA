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
from copy import deepcopy
import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.logger import get_logger
l = get_logger("make_shards")

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import argparse
from tqdm import tqdm
import os
import sys
import math
import numpy as np
import random
import cv2
from utils.common.files import clean_folder, read_json,is_json
from utils.datasets.slice_datasets import PatchDataset
from torchvision.transforms import transforms, RandomVerticalFlip, RandomHorizontalFlip

def apply_transform(images):
    '''
        apply tranformation functions on cv2 arrays (No aumenta datos)
    '''

    def apply_flip(images : dict, flip):
        if(random.random() > 0.5):
            for key in images.keys():
                images[key] = flip(p=1)(images[key])
        return images
    iter
    augment = transforms.Compose([
        lambda images : apply_flip(images,RandomVerticalFlip),
        lambda images : apply_flip(images,RandomHorizontalFlip),
        ])
    flipped = augment(images)
    return flipped

def apply_norm(pre_patch, post_patch, dis_id, tile_id, normalize, mean_stdv_json_path):
    '''
        apply transformation functions on cv2 arrays
    '''
    chips = {"pre": pre_patch, "post": post_patch}
    norm_chips = {}
    for prefix in ["pre", "post"]:
        curr_chip = np.array(chips[prefix]).astype(dtype='float64') / 255.0
        if normalize:
            is_json(mean_stdv_json_path)
            data_mean_stddev = read_json(mean_stdv_json_path)
            mean = data_mean_stddev[dis_id][tile_id][prefix]["mean"]
            mean_rgb = [mean[channel] for channel in ["R", "G", "B"]] 
            std = data_mean_stddev[dis_id][tile_id][prefix]["stdv"]
            std_rgb = [std[channel] for channel in ["R", "G", "B"]]
            norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_rgb, std=std_rgb)
            ])
            norm_chips[prefix] = norm(curr_chip).permute(1,2,0)
        else:
            norm_chips[prefix] = curr_chip
    return norm_chips["pre"], norm_chips["post"]

def shard_patches(dataset, split_name, mean_stddev_json, num_shards, out_path, transform = False, normalize = True):
    """
        Iterate through the dataset to produce shards of chips as numpy arrays, for imagery input and labels.
    """
    os.makedirs(out_path, exist_ok=True)  
    num_patches = len(dataset)
    # patch_per_shard
    pxs = math.ceil(num_patches / num_shards)
    shard_idx = [((i -1) * pxs, ((i) * pxs)) for i in range(1,num_shards+1)]
    for i, tpl in tqdm(enumerate(shard_idx),total=num_shards):
        begin_idx, end_idx = tpl
        # gets data
        image_patches = defaultdict(lambda :[])
        for j in range(begin_idx,end_idx):
            dis_id, tile_id, patch_id, data = dataset[j]
            image_patches["pre-orig"].append(deepcopy(data["pre_image"]))
            image_patches["post-orig"].append(deepcopy(data["post_image"]))
            if transform: data = apply_transform(**data)
            pre_img, post_img = apply_norm(data["pre_image"], data["post_image"], dis_id, tile_id, normalize, mean_stddev_json)            
            image_patches["pre-image"].append(pre_img) 
            image_patches["post-image"].append(post_img)
            image_patches["semantic-mask"].append(data["pre_mask"])
            image_patches["class-mask"].append(data["post_mask"]) 
    
        # save n shards
        for file_id, patch_list in image_patches.items():
            shard = np.stack(patch_list, axis=0)
            shard_path = os.path.join(out_path, f'{split_name}_{file_id}_{str(i).zfill(3)}.npy')
            np.save(shard_path, shard)
            l.info(f'Shape of last added shard to {f"{split_name}_shard"} list is {shard.shape}, dtype is {shard.dtype}.')
    
        #freeing memory
        del image_patches
        del shard

def create_shards(sliced_splits_json,mean_stddev_json,output_path,num_shards):
    
    def iterate_and_shard(split_name):
        l.info(f'Creating shards for {split_name} set ...')

        dataset = PatchDataset(split_name,sliced_splits_json)
        l.info(f'xBD_disaster_dataset {split_name} length: {len(dataset)}')

        clean_folder(output_path,split_name)
        out_dir = os.path.join(output_path,split_name)
        shard_patches(dataset, split_name, mean_stddev_json, num_shards,out_dir)

        l.info(f'Done creating shards for {split_name}')
    
    #could be parallelized
    iterate_and_shard("train")
    iterate_and_shard("val")

    l.info('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create shards from a sliced xBD dataset.')
    parser.add_argument(
        'sliced_splits_json',
        type=str,
        help=('Path to the json file with the train/val/test splits for sliced data.')
    )
    parser.add_argument(
        'mean_stddev_json_path',
        type=str,
        help=('Path to the json file with the mean and stdv.')
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help=('Path to folder for new sliced data.')
    )
    parser.add_argument(
       '-n','--num_shards',
        type=int,
        help=('Number of shards to be created for each file type.')
    )
    args = parser.parse_args()
    create_shards(args.sliced_splits_json, args.mean_stddev_json_path,args.output_path, args.num_shards)