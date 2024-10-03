# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import cv2
from torch.utils.data import Dataset
from utils.common.pathManager import FilePath


class TileDataset(Dataset):
    """`torch.utils.data.Dataset` class that implements the corresponding methods to access tiles
      from the raw xBD dataset folder."""

    def __init__(self, split_name: FilePath, splits_json_path: FilePath):
        self.split_name = split_name
        self.splits_json_path = splits_json_path

        splits_json_path.must_be_json()
        splits_all_disasters = splits_json_path.read_json()
        self.split_name = split_name
        data = splits_all_disasters[split_name]

        self.tile_list = [(dis_id, tile_id, tile)
                          for dis_id in data.keys()
                          for tile_id, tile in data[dis_id].items()]

    def __len__(self):
        return len(self.tile_list)

    def same_shape(self, dis_id, tile_id, img1, img2) -> bool:
        assert img1.shape[:2] == img2.shape[:2], \
            f'Images from {dis_id}_{tile_id} should be the same size, \
                  {img1.shape} != {img2.shape}.'
        return True

    def load_images(self, disaster_id, tile_id, tile):
        """Load images and mask from dataset paths"""
        data = {}
        data["pre_img"] = cv2.cvtColor(cv2.imread(
            tile["pre"]["image"]), cv2.COLOR_BGR2RGB)
        data["post_img"] = cv2.cvtColor(cv2.imread(
            tile["post"]["image"]), cv2.COLOR_BGR2RGB)
        data["bld_mask"] = cv2.imread(tile["pre"]["mask"])[:, :, 0]
        data["dmg_mask"] = cv2.imread(tile["post"]["mask"])[:, :, 0]

        self.same_shape(disaster_id, tile_id,
                        data["pre_img"], data["post_img"])
        self.same_shape(disaster_id, tile_id,
                        data["post_img"], data["bld_mask"])
        self.same_shape(disaster_id, tile_id,
                        data["post_img"], data["dmg_mask"])
        return data

    def __getitem__(self, i):
        disaster_id, tile_id, tile = self.tile_list[i]
        data = self.load_images(disaster_id, tile_id, tile)
        return disaster_id, tile_id, data
