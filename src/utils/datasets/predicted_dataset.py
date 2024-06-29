# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import numpy as np
from torch.utils.data import Dataset
import cv2
import os
import sys

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.pathManagers.predictedManager import PredictedPathManager
from utils.common.files import read_json, is_json

class PredictedDataset(Dataset):
    """Class that implements the corresponding methods to access raw xBD dataset data."""

    def __init__(self, split_raw_json_path : str, predicted_patches_folder_path : str):
        is_json(split_raw_json_path)
        splits_all_disasters = read_json(split_raw_json_path)
        tiles = splits_all_disasters["test"]
        
        predicted_dict = PredictedPathManager.load_paths(predicted_patches_folder_path)

        self.tile_list = [(dis_id, tile_id, tile, predicted_dict[dis_id][tile_id])
                          for dis_id in tiles.keys()
                          for tile_id, tile in tiles[dis_id].items()]

    def __len__(self):
        return len(self.tile_list)

    def same_shape(self, dis_id, tile_id, img1, img2) -> bool:
        assert img1.shape[:2] == img2.shape[:2], \
            f'Images from {dis_id}_{tile_id} should be the same size, {img1.shape} != {img2.shape}.'
        return True

    def load_images(self, tile : dict) -> dict:
        """Load images and mask from dataset paths"""
        data = {}
        data["pre_img"] = cv2.cvtColor(cv2.imread(tile["pre"]["image"]), cv2.COLOR_BGR2RGB)
        data["post_img"] = cv2.cvtColor(cv2.imread(tile["post"]["image"]), cv2.COLOR_BGR2RGB)
        data["bld_mask"] = cv2.imread(tile["pre"]["mask"])[:, :, 0]
        data["dmg_mask"] = cv2.imread(tile["post"]["mask"])[:, :, 0]
        data["bld_json"] = read_json(tile["pre"]["json"]) 
        data["dmg_json"] = read_json(tile["post"]["json"])
        return data

    def merge_patches(self, patch_path_dict : dict) -> np.ndarray:
        path_list = list(patch_path_dict.keys())
        path_list.sort()
        rows = []
        row = []
        for i, key in enumerate(path_list):
            mask = cv2.imread(patch_path_dict[key])[:, :, 0]
            row.append(mask)
            if(i%4==3):
                line = np.hstack(row)
                rows.append(line)
                row = []
        mask = np.vstack(rows)
        return mask
    
    def __getitem__(self, i : int) -> tuple[str, str, dict, np.ndarray]:
        disaster_id, tile_id, tile, patch_dict = self.tile_list[i]
        loaded_tile = self.load_images(tile)
        predicted_mask = self.merge_patches(patch_dict)
        return disaster_id, tile_id, loaded_tile, predicted_mask
