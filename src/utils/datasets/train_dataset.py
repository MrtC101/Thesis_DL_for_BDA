import math
import numpy as np
from torch import Tensor
import torch
from torchvision import transforms
from utils.common.files import read_json, is_json
from torch.utils.data import Dataset
import cv2
import os
import sys

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

class TrainDataset(Dataset):
    """Class that implements the corresponding methods to access raw xBD dataset data."""

    def __init__(self, split_name: str, splits_json_path: str, mean_stdv_json_path : str):
        self.split_name = split_name
        self.splits_json_path = splits_json_path

        is_json(splits_json_path)
        splits_all_disasters = read_json(splits_json_path)
        self.split_name = split_name
        data = splits_all_disasters[split_name]

        is_json(mean_stdv_json_path)
        self.data_mean_stddev = read_json(mean_stdv_json_path)

        self.tile_list = [(dis_id, tile_id, patch_id, files)
                          for dis_id in data.keys()
                          for tile_id in data[dis_id].keys()
                          for patch_id, files in data[dis_id][tile_id].items()]

    def __len__(self):
        return len(self.tile_list)

    def same_shape(self, dis_id, tile_id, img1, img2) -> bool:
        assert img1.shape[:2] == img2.shape[:2], \
            f'Images from {dis_id}_{tile_id} should be the same size, {img1.shape} != {img2.shape}.'
        return True

    def normalization(self, dis_id, tile_id, prefix : str, image : np.array) -> np.array:
        """A np.array of an image with format (w,h,c)."""
        norm_img = image.astype(np.float32) / 255.0
        
        mean = self.data_mean_stddev[dis_id][tile_id][prefix]["mean"]
        stdv = self.data_mean_stddev[dis_id][tile_id][prefix]["stdv"]

        # Compute mean and standard deviation for each channel
        mean_rgb = [mean[ch] for ch in ["R", "G", "B"]]
        std_rgb = [stdv[ch] for ch in ["R", "G", "B"]]

        # Define normalization steps
        normalization = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_rgb, std=std_rgb)
        ])

        # Apply normalization and permute channels
        normalized_img = normalization(norm_img)
        #normalized_img = normalized_img.permute(1, 2, 0) #Permute for visualization
        return normalized_img

    def __getitem__(self, i):
        disaster_id, tile_id, patch_id, patch = self.tile_list[i]
        data = {}
        data["pre_img"] = cv2.cvtColor(cv2.imread(patch["pre_img"]), cv2.COLOR_BGR2RGB)
        data["post_img"] = cv2.cvtColor(cv2.imread(patch["post_img"]), cv2.COLOR_BGR2RGB)
        data["bld_mask"] = cv2.imread(patch["bld_mask"])[:, :, 0].astype(np.uint8)
        data["dmg_mask"] = cv2.imread(patch["dmg_mask"])[:, :, 0].astype(np.uint8)
        data["pre_img"] = self.normalization(disaster_id, tile_id, "pre", data["pre_img"])
        data["post_img"] = self.normalization(disaster_id, tile_id, "post",data["post_img"])

        #Clean 5 label unidentified from data
        mask = data["bld_mask"] == 5
        data["bld_mask"][mask] = 0
        data["bld_mask"] = torch.from_numpy(data["bld_mask"]).long()

        mask = data["dmg_mask"] == 5
        data["dmg_mask"][mask] = 0
        data["dmg_mask"] = torch.from_numpy(data["dmg_mask"]).long()
        return disaster_id, tile_id, patch_id, data
    
    def get_sample_images(self, num_chips_to_viz):
        """
        Get a deterministic set of images in the specified set (train or val) by using the
          dataset and not the dataloader. Only works if the dataset is not IterableDataset.

        Returns:
            samples: a list of 'num_chips_to_viz` for visualization.
        """
        num_to_skip = 1  # first few chips might be mostly blank
        assert len(self) > num_to_skip + num_chips_to_viz

        keep_every = math.floor((len(self) - num_to_skip) / num_chips_to_viz)
        samples_idx_list = []

        for sample_idx in range(num_to_skip, len(self), keep_every):
            samples_idx_list.append(sample_idx)

        return samples_idx_list
    

    def save_pred_patch(self, pred_dmg_mask: Tensor, batch_idx, dis_id,tile_id,
                         patch_id, save_path: str) -> None:
        """Saves current prediction image"""
        os.makedirs(save_path, exist_ok=True)
        file = os.path.join(save_path, f"{batch_idx}_{dis_id}_{tile_id}_{patch_id}_dmg_mask.png")
        img = np.array(pred_dmg_mask.astype(np.uint8).permute(2,1,0))
        cv2.imwrite(file, img)
