# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
from utils.common.logger import get_logger
l = get_logger("make_smaller_tiles")

import argparse
import random
import numpy as np
import math
import cv2

from tqdm import tqdm
from os.path import join
from torchvision import transforms
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor

from utils.common.files import read_json
from utils.datasets.raw_datasets import TileDataset

def slice_tile(n, pre_image, post_image, pre_mask, post_mask):

    tile_h, tile_w = pre_image.shape[:2]

    assert tile_h % n == 0 and n > 0, f"Can't crop image into {n}x{n} equal parts."

    h_idx = [math.floor(tile_h*p) for p in np.arange(0, 1, 0.25)]
    w_idx = [math.floor(tile_w*p) for p in np.arange(0, 1, 0.25)]
    # l.info(f"{h_idx}")

    patch_h = math.floor(tile_h / n)
    patch_w = math.floor(tile_w / n)

    imgs = [pre_image, post_image, pre_mask, post_mask]
    keys = ["pre-image", "post-image", "semantic-mask", "class-mask"]

    def create_crop(i, j):
        crop = transforms.Compose([
            lambda img: img[i:i+patch_h, j:j+patch_w],
            # transforms.ToTensor()
        ])
        return crop

    def create_patch(patch_list, crop_transform):
        patch_dict = {}
        for key, img in zip(keys, imgs):
            patch = crop_transform(img)
            patch_dict[key] = patch
        patch_list.append(patch_dict)
        return patch_list

    patch_list = []
    for i in h_idx:
        for j in w_idx:
            crop_transform = create_crop(i, j)
            create_patch(patch_list, crop_transform)

    # pick 4 random slices from each tile
    for _ in range(0, 4):
        i = random.randint(5, h_idx[-1]-5)
        j = random.randint(5, w_idx[-1]-5)
        crop_transform = create_crop(i, j)
        create_patch(patch_list, crop_transform)

    return patch_list

def save_patches(disaster_id, tile_id, patch_list, sliced_path, split_name):
    split_folder = join(sliced_path, split_name)
    os.makedirs(split_folder, exist_ok=True)
    for i, patch in enumerate(patch_list):
        patch_id = f"{disaster_id}_{tile_id}_{i}"
        patch_folder = join(split_folder, patch_id)
        os.makedirs(patch_folder, exist_ok=True)
        for key in patch.keys():
            img_name = f"{patch_id}_{key}.png"
            path = join(patch_folder, img_name)
            cv2.imwrite(path, patch[key])

def slice_dataset(splits_json_path, output_path, batch_size):

    def iterate_and_slice(split_name):
        l.info(f'Starting slicing for {split_name}')
        dataset = TileDataset(split_name, splits_json_path)
        l.info(f'{split_name} dataset length before cropping: {len(dataset)}.')
        for dis_id, tile_id, data in tqdm(iter(dataset)):
            patch_list = slice_tile(4, **data)
            save_patches(dis_id, tile_id, patch_list,
                         output_path, split_name)
        l.info(f'{split_name} dataset length after cropping: {len(dataset)}.')
        l.info(f'Done slicing for {split_name}')

    iterate_and_slice("train")
    with ThreadPoolExecutor(max_workers=2) as executor:
        for split_name in ["train", "val"]:
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
        '-b', '--batch_size',
        type=int,
        default=1,
        help=('Size of the batch of images for augmentation.')
    )
    args = parser.parse_args()
    slice_dataset(args.split_json_path, args.output_dir, args.batch_size)
