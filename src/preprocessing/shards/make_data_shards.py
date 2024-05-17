from PIL import Image
from utils.common.files import clean_folder, dump_json, read_json, is_json
from torchvision.transforms import transforms, RandomVerticalFlip, \
    RandomHorizontalFlip
from utils.datasets.shard_datasets import ShardDataset
from utils.datasets.slice_datasets import PatchDataset
import random
import numpy as np
import math
from tqdm import tqdm
import argparse
from collections import defaultdict
from copy import deepcopy
# from concurrent.futures import ThreadPoolExecutor
import os
import sys
from utils.common.logger import LoggerSingleton

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
log = LoggerSingleton()


def apply_transform(images: dict) -> dict:
    """Apply tranformation functions on cv2 arrays.
      (IS NOT DATA AGUMENTATION)"""
    def apply_flip(images: dict, flip):
        if (random.random() > 0.5):
            for key in images.keys():
                images[key] = np.array(flip(p=1)(Image.fromarray(images[key])))
        return images
    augment = transforms.Compose([
        lambda images: apply_flip(images, RandomVerticalFlip),
        lambda images: apply_flip(images, RandomHorizontalFlip),
    ])
    flipped = augment(images)
    return flipped


def apply_norm(pre_image: np.array, post_image: np.array, dis_id: str,
               tile_id: str, normalize: bool, mean_stdv_json_path: str
               ) -> tuple[np.array]:
    """Apply transformation functions on cv2 arrays."""
    chips = {"pre": pre_image, "post": post_image}
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
            norm_chips[prefix] = norm(curr_chip).permute(1, 2, 0)
        else:
            norm_chips[prefix] = curr_chip
    return norm_chips["pre"], norm_chips["post"]


def shard_patches(dataset: PatchDataset, split_name: str,
                  mean_stddev_json: dict, num_shards: int, out_path,
                  transform=False, normalize=True) -> list[tuple[int]]:
    """Iterate through the dataset to produce shards of chips(patches) as a new
      `num_shards` numpy arrays.

      Args:
        dataset: PatchDataset that allows access to the sliced data.
        split_name: name of the current split ("train","val")
        mean_stddev_json: The loaded dictionary that is used for data
        normalization.
        num_shards: number of shards that will be created for this split.
        out_path: Path where the new data shards will be saved.
        transform: Boolean for applying transformation for this shard images
        normalize: Boolean for applying normalization for this shard images.
      Return:
        list[tuple[int]]: A list that stores the start and end patch index for
        each shard created.
    """
    os.makedirs(out_path, exist_ok=True)
    num_patches = len(dataset)
    # patch_per_shard
    pxs = math.ceil(num_patches / num_shards)
    shard_idxs = [((i - 1) * pxs, ((i) * pxs)) for i in range(1, num_shards+1)]
    for i, tpl in tqdm(enumerate(shard_idxs), total=num_shards):
        begin_idx, end_idx = tpl
        # gets data
        image_patches = defaultdict(lambda: [])
        for idx in range(begin_idx, end_idx):
            dis_id, tile_id, patch_id, data = dataset[idx]
            image_patches["pre-orig"].append(deepcopy(data["pre_image"]))
            image_patches["post-orig"].append(deepcopy(data["post_image"]))

            # transformations
            if transform:
                data = apply_transform(data)

            # color normalization
            pre_img, post_img = apply_norm(
                data["pre_image"], data["post_image"], dis_id, tile_id,
                normalize, mean_stddev_json)

            # replace non-classified pixels with background
            dmg_mask = np.where(data["post_mask"] == 5, 0, data["post_mask"])

            image_patches["pre-image"].append(pre_img)
            image_patches["post-image"].append(post_img)
            image_patches["semantic-mask"].append(data["pre_mask"])
            image_patches["class-mask"].append(dmg_mask)

        # save n shards
        ShardDataset.save_shard(image_patches, out_path, log, split_name, i)

        # freeing memory
        del image_patches

    return shard_idxs


def create_shards(sliced_splits_json: str, mean_stddev_json: str,
                  output_path: str, num_shards: int) -> None:
    """Stores each patch or chip from tiles in a large numpy array, so they can
    all be used as one large file. The `train` and `val` splits will be stored
    separately to distinguish them.

    Args:
        sliced_splits_json: Path to the JSON file that stores the sliced
          tiles patches split.
        mean_stddev_json: Path to the JSON file that stores the mean
        and standard deviation for each tile.
        output_path: Path to the folder where will be saved the new data
          shards.
        num_shards: number of shard to create from one split.
    """

    log.name = "Create data shards"

    def iterate_and_shard(split_name):
        log.info(f'Creating shards for {split_name} set ...')

        dataset = PatchDataset(split_name, sliced_splits_json)
        log.info(f'xBD_disaster_dataset {split_name} length: {len(dataset)}')

        clean_folder(output_path, split_name)
        out_dir = os.path.join(output_path, split_name)
        shard_idxs = shard_patches(
            dataset, split_name, mean_stddev_json,
            num_shards, out_dir,
            transform=False if split_name == "test" else True,
            normalize=False if split_name == "test" else True)

        idx_path = os.path.join(output_path, f"{split_name}_shard_idxs.json")
        dump_json(idx_path, {"shard_idxs": shard_idxs})

        log.info(f'Done creating shards for {split_name}')

    # could be parallelized
    iterate_and_shard("train")
    iterate_and_shard("val")
    iterate_and_shard("test")

    log.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create shards from a sliced xBD dataset.')
    parser.add_argument(
        'sliced_splits_json',
        type=str,
        help=('Path to the json file with the train/val/test splits for\
               sliced data.')
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
        '-n', '--num_shards',
        type=int,
        help=('Number of shards to be created for each file type.')
    )
    args = parser.parse_args()
    create_shards(args.sliced_splits_json, args.mean_stddev_json_path,
                  args.output_path, args.num_shards)
