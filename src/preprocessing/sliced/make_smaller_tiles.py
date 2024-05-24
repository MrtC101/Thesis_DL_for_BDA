# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Modificaciones (c) 2024 Martín Cogo Belver.
# Martín Cogo Belver has rights reserved over this modifications.
#
# Modification Notes:
# - Documentation added with docstrings for code clarity.
# - Re-implementation of methods to enhance readability and efficiency.
# - Re-implementation of features for improved functionality.
# - Changes in the logic of implementation for better performance.
# - Bug fixes in the code.
#
# See the LICENSE file in the root directory of this project for the full text of the MIT License.
import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from typing import Dict, List
from utils.datasets.raw_datasets import TileDataset
from utils.datasets.slice_datasets import PatchDataset
from utils.common.files import clean_folder
from torchvision import transforms
from tqdm import tqdm
import math
import numpy as np
import random
import argparse
from utils.common.logger import LoggerSingleton
log = LoggerSingleton()


def slice_tile(n: int, random_c: bool, pre_img: np.ndarray, post_img: np.ndarray,
               bld_mask: np.ndarray, dmg_mask: np.ndarray
               ) -> List[Dict[str, np.ndarray]]:
    """Slices each tile into `n` equal parts and creates patches.

    Args:
        n: Number of equal parts to slice the tile into.
        random_c: if create 4 more patches with random crop.
        pre_img: Pre-disaster image.
        post_img: Post-disaster image.
        bld_mask: Pre-disaster semantic mask.
        dmg_mask: Post-disaster class mask.

    Returns:
        List[Dict[str, np.ndarray]]: List of dictionaries containing patches
          for each image.
    """
    tile_h, tile_w = pre_img.shape[:2]

    assert tile_h % n == 0 and n > 0, \
        f"Can't crop image into {n}x{n} equal parts."

    h_idx = [math.floor(tile_h*p) for p in np.arange(0, 1, 0.25)]
    w_idx = [math.floor(tile_w*p) for p in np.arange(0, 1, 0.25)]
    # log.info(f"{h_idx}")

    patch_h = math.floor(tile_h / n)
    patch_w = math.floor(tile_w / n)

    imgs = [pre_img, post_img, bld_mask, dmg_mask]
    keys = ["pre-img", "post-img", "bld-mask", "dmg-mask"]

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

    if (random_c):
        # pick 4 random slices from each tile
        for _ in range(0, 4):
            i = random.randint(5, h_idx[-1]-5)
            j = random.randint(5, w_idx[-1]-5)
            crop_transform = create_crop(i, j)
            create_patch(patch_list, crop_transform)

    return patch_list


def slice_dataset(splits_json_path: str, output_path: str) -> None:
    """Slices each tile into 20 patches of the same size.

    Args:
        splits_json_path: Path to the JSON file that contains the dictionary
        that represents the dataset split.
        out_path: Path to the folder where the new patches will be stored.
    Example:
        >>> slice_dataset("data/xBD/splits/raw_splits.json","data/xBD/sliced")
    """

    log.name = "Create data patches (chips)"

    def iterate_and_slice(split_name):
        log.info(f'Starting slicing for {split_name}')

        dataset = TileDataset(split_name, splits_json_path)
        num_tile = len(dataset)
        log.info(f'{split_name} dataset length before cropping: {num_tile}.')

        clean_folder(output_path, split_name)
        split_folder = os.path.join(output_path, split_name)

        for dis_id, tile_id, data in tqdm(iter(dataset), total=num_tile):
            patch_list = slice_tile(4, split_name != "test", **data)
            PatchDataset.save_patches(dis_id, tile_id, patch_list, split_folder)
        length = 16 * num_tile if split_name == "test" else 20 * num_tile
        log.info(f'Done slicing for {split_name}, length after cropping: {length}.')

    # could be parallelized
    iterate_and_slice("train")
    iterate_and_slice("val")
    iterate_and_slice("test")

    log.info('Done')


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
    args = parser.parse_args()
    slice_dataset(args.split_json_path, args.output_dir)
