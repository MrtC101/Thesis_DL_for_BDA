# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys
import cv2

from utils.common.pathManager import FilePath

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """`torch.utils.data.Dataset` class that implements the corresponding methods to access tiles
      from a JSON file with Tiles from the xBD dataset."""

    def __init__(self, split_name: FilePath, splits_json_path: FilePath):
        self.split_name = split_name
        self.splits_json_path = splits_json_path

        splits_json_path.must_be_json()
        splits_all_disasters = splits_json_path.read_json()
        self.split_name = split_name
        data = splits_all_disasters[split_name]

        self.tile_list = [(dis_id, tile_id, patch_id, files)
                          for dis_id in data.keys()
                          for tile_id in data[dis_id].keys()
                          for patch_id, files in data[dis_id][tile_id].items()]

    def __len__(self):
        return len(self.tile_list)

    def __getitem__(self, i):
        disaster_id, tile_id, patch_id, patch = self.tile_list[i]
        data = self.load_patches(disaster_id, tile_id, patch_id, patch)
        return disaster_id, tile_id, patch_id, data

    def load_patches(self, disaster_id, tile_id, patch_id, patch):
        """
        Loads patch data from disk.

        Args:
            disaster_id (str): The ID of the disaster.
            tile_id (str): The ID of the tile.
            patch_id (str): The ID of the patch.
            patch (dict): A dictionary containing file paths to patch images.

        Returns:
            dict: A dictionary containing loaded patch data.
        """
        data = {}
        data["pre_img"] = cv2.cvtColor(
            cv2.imread(patch["pre_img"]), cv2.COLOR_BGR2RGB)
        data["post_img"] = cv2.cvtColor(
            cv2.imread(patch["post_img"]), cv2.COLOR_BGR2RGB)
        data["bld_mask"] = cv2.imread(patch["bld_mask"])[:, :, 0]
        data["dmg_mask"] = cv2.imread(patch["dmg_mask"])[:, :, 0]
        return data

    @staticmethod
    def save_patches(disaster_id, tile_id, patch_list,
                     split_folder: FilePath):
        """
        Saves a list of patches to disk.

        Args:
            disaster_id (str): The ID of the disaster.
            tile_id (str): The ID of the tile.
            patch_list (list): A list of patches to save.
            split_folder (str): The directory to save the patches to.
        """
        for i, patch in enumerate(patch_list):
            patch_id = f"{disaster_id}_{tile_id}_{str(i).zfill(3)}"
            patch_folder = split_folder.join(patch_id)
            patch_folder.create_folder()
            for key in patch.keys():
                img_name = f"{patch_id}_{key}.png"
                path = patch_folder.join(img_name)
                if (key in ['pre-img', 'post-img',]):
                    new_patch = cv2.cvtColor(patch[key], cv2.COLOR_RGB2BGR)
                elif (key in ['bld-mask', 'dmg-mask']):
                    new_patch = patch[key]
                cv2.imwrite(path, new_patch)
