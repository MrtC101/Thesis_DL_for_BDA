# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
from collections import OrderedDict
import os
import sys
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset

from utils.common.pathManager import FilePath

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))


class TrainDataset(Dataset):
    """`torch.utils.data.Dataset` class that implements the corresponding methods to access cropped
    patches from  from the xBD dataset."""

    def __init__(self, split_name: str, splits_json_path: FilePath,
                 mean_stdv_json_path: FilePath):
        self.split_name = split_name
        self.splits_json_path = splits_json_path
        splits_json_path.must_be_json()
        splits_all_disasters = splits_json_path.read_json()
        data = splits_all_disasters[split_name]

        mean_stdv_json_path.must_be_json()
        self.data_mean_stddev = mean_stdv_json_path.read_json()

        self.tile_list = [(dis_id, tile_id, patch_id, files)
                          for dis_id in data.keys()
                          for tile_id in data[dis_id].keys()
                          for patch_id, files in data[dis_id][tile_id].items()]

        self.patch_list = [(dis_id, tile_id, tile_dict)
                           for dis_id in data.keys()
                           for tile_id, tile_dict in data[dis_id].items()]

        self.normalize = True

    def __len__(self):
        return len(self.tile_list)

    def _normalization(self, statistic: dict, prefix: str,
                       image: torch.Tensor) -> torch.Tensor:
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

    def set_normalize(self, normalize: bool) -> None:
        self.normalize = normalize

    def __getitem__(self, i: int) -> tuple:
        """
            Returns a tuple (disaster_id, tile_id, patch_id, data)
        """
        disaster_id, tile_id, patch_id, patch = self.tile_list[i]

        pre_img = read_image(patch["pre_img"])
        post_img = read_image(patch["post_img"])
        bld_mask = read_image(patch["bld_mask"]).squeeze(0)
        dmg_mask = read_image(patch["dmg_mask"]).squeeze(0)

        if (self.normalize):
            tile_stat_dict = self.data_mean_stddev[disaster_id][tile_id]
            pre_img = self._normalization(tile_stat_dict, "pre", pre_img)
            post_img = self._normalization(tile_stat_dict, "post", post_img)

        # Clean 5 label unidentified from data
        bld_mask[bld_mask == 5] = 0
        dmg_mask[dmg_mask == 5] = 0

        bld_mask = bld_mask.to(torch.int64)
        dmg_mask = dmg_mask.to(torch.int64)

        data = {
            "pre_img": pre_img,
            "post_img": post_img,
            "bld_mask": bld_mask,
            "dmg_mask": dmg_mask,
        }
        return disaster_id, tile_id, patch_id, data

    def get_by_id(self, i):
        disaster_id, tile_id, tile_dict = self.patch_list[i]
        keys_list = sorted(int(k) for k in tile_dict.keys())
        data = OrderedDict()
        for patch_id in keys_list:
            k = str(patch_id).zfill(3)
            patch = tile_dict[k]

            # Leer imágenes y máscaras
            pre_img = read_image(patch["pre_img"])
            post_img = read_image(patch["post_img"])
            bld_mask = read_image(patch["bld_mask"]).squeeze(0)
            dmg_mask = read_image(patch["dmg_mask"]).squeeze(0)

            tile_stat_dict = self.data_mean_stddev[disaster_id][tile_id]
            pre_img = self._normalization(tile_stat_dict, "pre", pre_img)
            post_img = self._normalization(tile_stat_dict, "post", post_img)

            # Ajustar las máscaras para asegurarse de que 5 sea convertido a 0
            bld_mask[bld_mask == 5] = 0
            dmg_mask[dmg_mask == 5] = 0
            bld_mask = bld_mask.to(torch.int64)
            dmg_mask = dmg_mask.to(torch.int64)

            # Almacenar los datos procesados en el diccionario
            data[str(patch_id)] = {
                "pre_img": pre_img,
                "post_img": post_img,
                "bld_mask": bld_mask,
                "dmg_mask": dmg_mask,
            }

        return disaster_id, tile_id, data
