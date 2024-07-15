# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import math
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from torchvision import transforms
from utils.common.pathManager import FilePath
from utils.datasets.raw_datasets import TileDataset
from utils.datasets.slice_datasets import PatchDataset
from utils.loggers.console_logger import LoggerSingleton

log = LoggerSingleton()


def slice_tile(n: int, random_c: bool, pre_img: np.ndarray,
               post_img: np.ndarray, bld_mask: np.ndarray,
               dmg_mask: np.ndarray) -> List[Dict[str, np.ndarray]]:
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

    return patch_list


def slice_dataset(splits_json_path: FilePath, output_path: FilePath) -> None:
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

        output_path.clean_folder(split_name)
        split_folder = output_path.join(split_name)

        for dis_id, tile_id, data in tqdm(iter(dataset), total=num_tile):
            patch_list = slice_tile(4, split_name != "test", **data)
            PatchDataset.save_patches(
                dis_id, tile_id, patch_list, split_folder)
        length = 16 * num_tile
        log.info(
            f'Done slicing for {split_name}, length after cropping: {length}.')

    splits_all_disasters = splits_json_path.read_json()
    for setName in splits_all_disasters.keys():
        # could be parallelized
        iterate_and_slice(setName)

    log.info('Done')
