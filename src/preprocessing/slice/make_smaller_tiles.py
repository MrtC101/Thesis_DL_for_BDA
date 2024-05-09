# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import sys
if(os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.visualization.logger import get_logger
l = get_logger("Slice_Dataset")

import cv2
import math
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader
from utils.common.files import read_json
from utils.datasets.inference_datasets import SliceDataset


def slice_tile(self, n, pre_img, post_img, pre_mask, post_mask):

    tile_h, tile_w = pre_img.shape[:2]

    assert tile_h % n == 0 and n > 0, f"Can't crop image into {n}x{n} equal parts."

    h_idx = [math.floor(tile_h*p) for p in np.arange(0, 1, 0.25)]
    w_idx = [math.floor(tile_w*p) for p in np.arange(0, 1, 0.25)]
    #l.info(f"{h_idx}")
    
    patch_h = math.floor(tile_h / n)
    patch_w = math.floor(tile_w / n)

    imgs = [pre_img, post_img, pre_mask, post_mask]
    keys = ["pre-img", "post-img", "semantic-mask", "class-mask"]

    def create_crop(i, j):
        crop = transforms.Compose([
            lambda img: img[i:i+patch_h, j:j+patch_w],
            #transforms.ToTensor()
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

def create_dataset(set_name,splits_json,output_path):   
    l.info(f'{set_name} dataset length before cropping: {set_length}.')

    dataset = SliceDataset(set_name,split,output_path)

    l.info(f'{set_name} dataset length after cropping: {len(dataset)}.')
    return dataset

def slice_dataset(splits_json, output_path, batch_size):
    
    def iterate_and_slice(split_name):
        l.info(f'Starting slicing for {split_name}')      
        dataset = create_dataset(split_name,splits_json, output_path)
        dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=8)
        
        for disaster_id,tile_id,data in tqdm(dataloader):
            pre_img = data["pre_image"]
            post_img = data["post_image"]
            pre_mask = data["semantic_mask"]
            post_mask = data["class_mask"]
            patch_list = slice_tile(
            4,pre_img, post_img, pre_mask, post_mask)

            def save_patches()
                split_folder = join(self.augmented_path,self.split_name)
                os.makedirs(split_folder,exist_ok=True)
                for i,patch in enumerate(patch_list):
                    patch_id = f"{disaster_id}_{tile_id}_{i}"
                    patch_folder = join(split_folder,patch_id)
                    os.makedirs(patch_folder, exist_ok=True)
                    for key in patch.keys():
                        img_name = f"{patch_id}_{key}.png"
                        path = join(patch_folder,img_name)
                        cv2.imwrite(path,patch[key])


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