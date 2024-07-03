# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys
import math
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.files import read_json, is_json

class TrainDataset(Dataset):
    """Class that implements the corresponding methods to access raw xBD dataset data."""

    def __init__(self, split_name: str, splits_json_path: str, mean_stdv_json_path : str):
        self.split_name = split_name
        self.splits_json_path = splits_json_path
        is_json(splits_json_path)
        splits_all_disasters = read_json(splits_json_path)
        data = splits_all_disasters[split_name]

        is_json(mean_stdv_json_path)
        self.data_mean_stddev = read_json(mean_stdv_json_path)

        self.tile_list = [(dis_id, tile_id, patch_id, files)
                          for dis_id in data.keys()
                          for tile_id in data[dis_id].keys()
                          for patch_id, files in data[dis_id][tile_id].items()]
        self.normalize=True

    def __len__(self):
        return len(self.tile_list)

    def _normalization(self, statistic : dict, prefix : str, image : torch.Tensor) -> torch.Tensor:
        """A np.array of an image with format (w,h,c)."""
        
        mean = statistic[prefix]["mean"]
        stdv = statistic[prefix]["stdv"]

        # Compute mean and standard deviation for each channel
        mean_rgb = [mean[ch] for ch in ["R", "G", "B"]]
        std_rgb = [stdv[ch] for ch in ["R", "G", "B"]]

        # Define normalization steps
        normalization = transforms.Compose([
            lambda img: img.to(torch.float32) / 255.0,
            transforms.Normalize(mean=mean_rgb, std=std_rgb)
        ])

        # Apply normalization
        norm_img = normalization(image)
        return norm_img

    def set_normalize(self, normalize : bool) -> None:
        self.normalize = normalize

    def __getitem__(self, i : int) -> tuple:
        disaster_id, tile_id, patch_id, patch = self.tile_list[i]
        
        pre_img = read_image(patch["pre_img"])
        post_img = read_image(patch["post_img"])
        bld_mask = read_image(patch["bld_mask"]).squeeze(0)
        dmg_mask = read_image(patch["dmg_mask"]).squeeze(0)

        if(self.normalize):
            tile_stat_dict = self.data_mean_stddev[disaster_id][tile_id]
            pre_img = self._normalization(tile_stat_dict, "pre", pre_img)
            post_img = self._normalization(tile_stat_dict, "post", post_img)
           
        #Clean 5 label unidentified from data
        bld_mask[bld_mask == 5] = 0 
        dmg_mask[dmg_mask == 5] = 0 

        bld_mask = bld_mask.to(torch.int64)
        dmg_mask = dmg_mask.to(torch.int64)

        data = {
            "pre_img" : pre_img,
            "post_img" : post_img,
            "bld_mask" : bld_mask,
            "dmg_mask" : dmg_mask,
        }        
        return disaster_id, tile_id, patch_id, data   