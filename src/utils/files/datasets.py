# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.logger import get_logger
l = get_logger("delete_extra")

import math
import torch
import random
import cv2
from os.path import join
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.files.common import read_json, is_dir

"""Deletable"""
import numpy as np
from PIL import Image
from glob import glob
from skimage import transform


def same_shape(dis_id,tile_id,img1, img2) -> bool:
    assert img1.shape[:2] == img2.shape[:2], \
        f'Images from {dis_id}_{tile_id} should be the same size, {img1.shape} != {img2.shape}.'
    return True


class TilesDataset(Dataset):
    """
        Dataset that uses a dictionary to manage files path that's builded with the data structure DisasterDict from path_manager.py
    """

    def __init__(self, data: dict):

        self.tile_list = [(dis_id, tile_id, tile)
                          for dis_id in data.keys()
                          for tile_id, tile in data[dis_id].items()]

    def __len__(self):
        return len(self.tile_list)

    def __getitem__(self, i):

        disaster_id, tile_id, tile = self.tile_list[i]

        pre_img = cv2.imread(tile["pre"]["image"], cv2.COLOR_BGR2RGB)
        post_img = cv2.imread(tile["post"]["image"], cv2.COLOR_BGR2RGB)
        post_json = read_json(tile["post"]["json"])
        pre_json = read_json(tile["pre"]["json"])

        return {
            "dis_id": disaster_id,
            "tile_id": tile_id,
            "pre_img": pre_img,
            "post_img": post_img,
            "pre_json": pre_json,
            "post_json": post_json
        }

class SliceDataset(Dataset):

    def __init__(self, split_name: str,split_dict: dict, augmented_path: str):
        """
        Args:
            split_dict: Dict builded with DisasterDict loaded from json file.
            augmented_path: Path to the directory were augmented dataset will be created
        """
        try:
            is_dir(augmented_path)
        except AssertionError as e:
            l.critical(e)

        self.out_path = augmented_path
        self.split_name = split_name
        self.tile_list = [(dis_id, tile_id, tile)
                          for dis_id in split_dict.keys()
                          for tile_id, tile in split_dict[dis_id].items()]

    def __len__(self):
        return len(self.tile_list)

    @classmethod
    def slice_tile(self, n, pre_img, post_img, pre_mask, post_mask):

        tile_h, tile_w = pre_img.shape[:2]

        assert tile_h % n == 0 and n > 0, f"Can't crop image into {n}x{n} equal parts."

        h_idx = [math.floor(tile_h*p) for p in np.arange(0, 1, 0.25)]
        w_idx = [math.floor(tile_w*p) for p in np.arange(0, 1, 0.25)]
        #l.info(f"{h_idx}")
        
        patch_h = math.floor(tile_h / n)
        patch_w = math.floor(tile_w / n)

        imgs = [pre_img, post_img, pre_mask, post_mask]
        keys = ["pre_img", "post_img", "semantic_mask", "class_mask"]

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

    def __getitem__(self, i):

        disaster_id, tile_id, tile = self.tile_list[i]

        pre_img = cv2.imread(tile["pre"]["image"], cv2.COLOR_BGR2RGB)
        post_img = cv2.imread(tile["post"]["image"], cv2.COLOR_BGR2RGB)
        # post_json = read_json(tile["post"]["json"])
        # pre_json = read_json(tile["pre"]["json"])
        pre_mask = cv2.imread(tile["pre"]["mask"], cv2.COLOR_BGR2RGB)
        post_mask = cv2.imread(tile["post"]["mask"], cv2.COLOR_BGR2RGB)

        same_shape(disaster_id,tile_id,pre_img, post_img)
        same_shape(disaster_id,tile_id,pre_img, pre_mask)
        same_shape(disaster_id,tile_id,post_img, post_mask)

        patch_list = self.slice_tile(
            4,pre_img, post_img, pre_mask, post_mask)

        split_folder = join(self.out_path,self.split_name)
        os.makedirs(split_folder,exist_ok=True)

        for i,patch in enumerate(patch_list):
            patch_id = f"{disaster_id}_{tile_id}_{i}"
            patch_folder = join(split_folder,patch_id)
            os.makedirs(patch_folder, exist_ok=True)
            for key in patch.keys():
                img_name = f"{patch_id}_{key}.png"
                path = join(patch_folder,img_name)
                cv2.imwrite(path,patch[key])

        return {
            'pre_image': TF.to_tensor(pre_img),
            'post_image': TF.to_tensor(post_img),
            'semantic_mask': TF.to_tensor(pre_mask),
            'class_mask': TF.to_tensor(post_mask)
        }

class ShardDataset(DisasterDataset):

    shard_names = ["pre_image_chips",
                   "post_image_chips",
                   "bld_mask_chips",
                   "dmg_mask_chips",
                   "pre_img_tiles",
                   "post_img_tiles"]

    def __init__(self, sliced_data_dict, i_shard, set_name, data_mean_stddev, transform: bool, normalize: bool):

        self.data = sliced_data_dict

        self.do_transform: bool = transform
        self.do_normalize: bool = normalize
        self.data_mean_stddev = data_mean_stddev
        self.shards = {}

        path_list = [
            join(data_dir, f'{set_name}_{shard_name}_{i_shard}.npy')
            for shard_name in self.shard_names
        ]
        for i, shard_path in enumerate(path_list):
            self.shards[self.shard_names[i]] = np.load(shard_path)
            l.info(f'{path_list[i]} loaded {self.shard.shape}')

    def __len__(self):
        return len(self.pre_image_chip_shard)

    @classmethod
    def apply_transform(self, mask, pre_img, post_img, damage_class):
        '''
        apply tranformation functions on cv2 arrays
        '''
        # Random horizontal flipping
        if random.random() > 0.5:
            mask = cv2.flip(mask, flipCode=1)
            pre_img = cv2.flip(pre_img, flipCode=1)
            post_img = cv2.flip(post_img, flipCode=1)
            damage_class = cv2.flip(damage_class, flipCode=1)

        # Random vertical flipping
        if random.random() > 0.5:
            mask = cv2.flip(mask, flipCode=0)
            pre_img = cv2.flip(pre_img, flipCode=0)
            post_img = cv2.flip(post_img, flipCode=0)
            damage_class = cv2.flip(damage_class, flipCode=0)

        return mask, pre_img, post_img, damage_class

    def apply_norm(self, pre_chip, post_chip):
        '''
        apply tranformation functions on cv2 arrays
        '''
        chips = {"pre": pre_chip, "post": post_chip}
        norm_chips = {}
        for prefix in ["pre", "post"]:
            curr_chip = np.array(chips[prefix]).astype(dtype='float64') / 255.0
            if self.do_normalize:
                # normalize the images based on a tilewise mean & std dev
                img_tile_file = join(self.data_dir, prefix, "???")
                mean, stddev = self.data_mean_stddev[img_tile_file]
                norm = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=stddev)
                ])
                norm_chips[prefix] = norm(curr_chip)
            else:
                norm_chips[prefix] = curr_chip
        return norm_chips["pre"], norm_chips["post"]

    def __getitem__(self, i):

        pre_img: np.array = self.shards["pre_image_chips"][i]
        post_img = self.shards["post_image_chips"][i]
        mask = self.shards["bld_mask_chips"][i]
        damage_class = self.shards["dmg_mask_chips"][i]

        # copy original image for viz
        pre_img_orig = pre_img
        post_img_orig = post_img

        if self.do_transform:
            mask, pre_img, post_img, damage_class = self.apply_transform(
                mask, pre_img, post_img, damage_class)

        pre_img, post_img = self.apply_norm(pre_img, post_img)

        # convert eveything to arrays
        pre_img = np.array(pre_img)
        post_img = np.array(post_img)
        mask = np.array(mask)
        damage_class = np.array(damage_class)

        # replace non-classified pixels with background
        damage_class = np.where(damage_class == 5, 0, damage_class)

        return {
            'pre_image': torch.from_numpy(pre_img).type(torch.FloatTensor),
            'post_image': torch.from_numpy(post_img).type(torch.FloatTensor),
            'building_mask': torch.from_numpy(mask).type(torch.LongTensor),
            'damage_mask': torch.from_numpy(damage_class).type(torch.LongTensor),
            'pre_image_orig': transforms.ToTensor()(pre_img_orig),
            'post_image_orig': transforms.ToTensor()(post_img_orig)
        }

class TrainShardsDataset(Dataset):
    pass

class TestDataset(Dataset):
    pass

## DELETABLE?

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
class TestDataset(Dataset):

    @classmethod
    def apply_transform(self, mask, pre_img, post_img, damage_class):
        '''
        apply tranformation functions on PIL images 
        '''
        if random.random() > 0.5:
            # Resize
            img_h = pre_img.size[0]
            img_w = pre_img.size[1]

            resize = transforms.Resize(
                size=(int(round(1.016*img_h)), int(round(1.016*img_w))))
            mask = resize(mask)
            pre_img = resize(pre_img)
            post_img = resize(post_img)
            damage_class = resize(damage_class)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                pre_img, output_size=(img_h, img_w))
            mask = TF.crop(mask, i, j, h, w)
            pre_img = TF.crop(pre_img, i, j, h, w)
            post_img = TF.crop(post_img, i, j, h, w)
            damage_class = TF.crop(damage_class, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            mask = TF.hflip(mask)
            pre_img = TF.hflip(pre_img)
            post_img = TF.hflip(post_img)
            damage_class = TF.hflip(damage_class)

        # Random vertical flipping
        if random.random() > 0.5:
            mask = TF.vflip(mask)
            pre_img = TF.vflip(pre_img)
            post_img = TF.vflip(post_img)
            damage_class = TF.vflip(damage_class)

        return mask, pre_img, post_img, damage_class

    def __getitem__(self, i):
        imgs_dir = os.path.join(
            self.data_dir, self.dataset_sub_dir[i].replace('labels', 'images'))
        imgs_dir_tile = self.dataset_sub_dir[i].replace('labels', 'images')
        masks_dir = os.path.join(
            self.data_dir, self.dataset_sub_dir[i].replace('labels', 'targets_border2'))
        preds_dir = os.path.join(
            self.data_dir, self.dataset_sub_dir[i].replace('labels', 'predictions'))

        idx = imgs_dir

        img_suffix = '_' + imgs_dir.split('_')[-1]
        img_suffix_tile = '_' + imgs_dir_tile.split('_')[-1]
        mask_suffix = '_' + masks_dir.split('_')[-1]

        pre_img_tile_name = imgs_dir_tile[0:-1 *
                                          (len(img_suffix_tile))] + '_pre_disaster'
        pre_img_file_name = imgs_dir[0:-1 *
                                     (len(img_suffix))] + '_pre_disaster' + img_suffix
        pre_img_file = glob(pre_img_file_name + '.*')

        mask_file_name = masks_dir[0:-1*(len(mask_suffix))] + \
            '_pre_disaster_b2' + mask_suffix
        mask_file = glob(mask_file_name + '.*')

        post_img_tile_name = pre_img_tile_name.replace('pre', 'post')
        post_img_file_name = pre_img_file_name.replace('pre', 'post')
        post_img_file = glob(post_img_file_name + '.*')

        damage_class_file_name = mask_file_name.replace('pre', 'post')
        damage_class_file = glob(damage_class_file_name + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file_name}'
        assert len(pre_img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {pre_img_file_name}'
        assert len(post_img_file) == 1, \
            f'Either no post disaster image or multiple images found for the ID {idx}: {post_img_file_name}'
        assert len(damage_class_file) == 1, \
            f'Either no damage class image or multiple images found for the ID {idx}: {damage_class_file_name}'

        mask = Image.open(mask_file[0])
        pre_img = Image.open(pre_img_file[0])
        post_img = Image.open(post_img_file[0])
        damage_class = Image.open(damage_class_file[0])

        assert pre_img.size == mask.size, \
            f'Image and building mask {idx} should be the same size, but are {pre_img.size} and {mask.size}'
        assert pre_img.size == damage_class.size, \
            f'Image and damage classes mask {idx} should be the same size, but are {pre_img.size} and {damage_class.size}'
        assert pre_img.size == post_img.size, \
            f'Pre_ & _post disaster Images {idx} should be the same size, but are {pre_img.size} and {post_img.size}'

        if self.transform is True:
            mask, pre_img, post_img, damage_class = self.apply_transform(
                mask, pre_img, post_img, damage_class)

        # copy original image for viz
        pre_img_orig = pre_img
        post_img_orig = post_img

        if self.normalize is True:
            # normalize the images based on a tilewise mean & std dev --> pre_
            mean_pre = self.data_mean_stddev[os.path.join(
                self.data_dir, pre_img_tile_name)][0]
            stddev_pre = self.data_mean_stddev[os.path.join(
                self.data_dir, pre_img_tile_name)][1]
            norm_pre = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_pre, std=stddev_pre)
            ])
            pre_img = norm_pre(np.array(pre_img).astype(dtype='float64')/255.0)

            # normalize the images based on a tilewise mean & std dev --> post_
            mean_post = self.data_mean_stddev[os.path.join(
                self.data_dir, post_img_tile_name)][0]
            stddev_post = self.data_mean_stddev[os.path.join(
                self.data_dir, post_img_tile_name)][1]
            norm_post = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_post, std=stddev_post)
            ])
            post_img = norm_post(
                np.array(post_img).astype(dtype='float64')/255.0)

        # convert eveything to arrays
        pre_img = np.array(pre_img)
        post_img = np.array(post_img)
        mask = np.array(mask)
        damage_class = np.array(damage_class)

        # replace non-classified pixels with background
        damage_class = np.where(damage_class == 5, 0, damage_class)

        return {'pre_image': torch.from_numpy(pre_img).type(torch.FloatTensor), 'post_image': torch.from_numpy(post_img).type(torch.FloatTensor), 'building_mask': torch.from_numpy(mask).type(torch.LongTensor), 'damage_mask': torch.from_numpy(damage_class).type(torch.LongTensor), 'pre_image_orig': transforms.ToTensor()(pre_img_orig), 'post_image_orig': transforms.ToTensor()(post_img_orig), 'img_file_idx': imgs_dir[0:-1*(len(img_suffix))].split('/')[-1] + img_suffix, 'preds_img_dir': preds_dir}