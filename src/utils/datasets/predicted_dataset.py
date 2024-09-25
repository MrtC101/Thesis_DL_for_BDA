# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import cv2
import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.pathManager import FilePath
from utils.pathManagers.predictedManager import PredictedPathManager


class PredictedDataset(Dataset):
    """Class that implements the corresponding methods to access raw xBD
    dataset data."""

    def __init__(self, split_raw_json_path: FilePath,
                 predicted_patches_folder_path: FilePath):
        split_raw_json_path.is_json()
        splits_all_disasters = split_raw_json_path.read_json()
        tiles = splits_all_disasters["test"]

        predicted_dict = PredictedPathManager.load_paths(
            predicted_patches_folder_path)

        self.tile_list = [(dis_id, tile_id, tile, predicted_dict[dis_id][tile_id])
                          for dis_id in tiles.keys()
                          for tile_id, tile in tiles[dis_id].items()]

    def __len__(self):
        return len(self.tile_list)

    def same_shape(self, dis_id, tile_id, img1, img2) -> bool:
        assert img1.shape[:2] == img2.shape[:2], 'Images from' + \
            f'{dis_id}_{tile_id} should be the same size,' + \
            f'{img1.shape} != {img2.shape}.'
        return True

    @staticmethod
    def save_pred_patch(pred_dmg_mask: torch.Tensor | np.ndarray,
                        batch_idx: str, dis_idx: str, tile_idx: str, patch_idx: str,
                        save_path: str) -> None:
        """Saves current prediction image"""
        os.makedirs(save_path, exist_ok=True)

        for ch, (dis_id, tile_id, patch_id) in enumerate(zip(dis_idx, tile_idx, patch_idx)):
            file = os.path.join(save_path,
                                f"{dis_id}_{tile_id}_{patch_id}_dmg_mask.png")
            mask = torch.Tensor.cpu(pred_dmg_mask[ch])
            img = np.array(mask).astype(np.uint8)
            cv2.imwrite(file, img)

    def _load_images(self, tile: dict) -> dict:
        """Load images and mask from dataset paths"""
        data = {}
        data["pre_img"] = cv2.cvtColor(cv2.imread(tile["pre"]["image"]), cv2.COLOR_BGR2RGB)
        data["post_img"] = cv2.cvtColor(cv2.imread(tile["post"]["image"]), cv2.COLOR_BGR2RGB)
        data["bld_mask"] = cv2.imread(tile["pre"]["mask"])[:, :, 0]
        data["dmg_mask"] = cv2.imread(tile["post"]["mask"])[:, :, 0]
        if "json" in tile["pre"].keys():
            data["bld_json"] = FilePath(tile["pre"]["json"]).read_json()
        else:
            data["bld_json"] = None
        if "json" in tile["post"].keys():
            data["dmg_json"] = FilePath(tile["post"]["json"]).read_json()
        else:
            data["dmg_json"] = None
        data["pre_img"] = torch.Tensor(data["pre_img"]).to(torch.uint8)
        data["post_img"] = torch.Tensor(data["post_img"]).to(torch.uint8)
        data["bld_mask"] = torch.Tensor(data["bld_mask"]).to(torch.uint8)
        data["dmg_mask"] = torch.Tensor(data["dmg_mask"]).to(torch.uint8)
        return data

    def merge_patches(self, patch_path_dict: dict) -> np.ndarray:
        path_list = list(patch_path_dict.keys())
        path_list.sort()
        rows = []
        row = []
        for i, key in enumerate(path_list):
            mask = cv2.imread(patch_path_dict[key])[:, :, 0]
            row.append(mask)
            if (i % 4 == 3):
                line = np.hstack(row)
                rows.append(line)
                row = []
        mask = np.vstack(rows)
        return mask

    def __getitem__(self, i: int) -> tuple[str, str, dict, np.ndarray]:
        disaster_id, tile_id, tile, patch_dict = self.tile_list[i]
        loaded_tile = self._load_images(tile)
        predicted_mask = self.merge_patches(patch_dict)
        return disaster_id, tile_id, loaded_tile, torch.Tensor(predicted_mask)
