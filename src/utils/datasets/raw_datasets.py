from utils.common.files import read_json, is_json
from torch.utils.data import Dataset
import cv2
import os
import sys

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))


class TileDataset(Dataset):
    """Class that implements the corresponding methods to access raw xBD
    dataset data.
    """

    def __init__(self, split_name: str, splits_json_path: str):
        self.split_name = split_name
        self.splits_json_path = splits_json_path

        is_json(splits_json_path)
        splits_all_disasters = read_json(splits_json_path)
        self.split_name = split_name
        data = splits_all_disasters[split_name]

        self.tile_list = [(dis_id, tile_id, tile)
                          for dis_id in data.keys()
                          for tile_id, tile in data[dis_id].items()]

    def __len__(self):
        return len(self.tile_list)

    def same_shape(self, dis_id, tile_id, img1, img2) -> bool:
        assert img1.shape[:2] == img2.shape[:2], \
            f'Images from {dis_id}_{tile_id} should be the same size,\
                  {img1.shape} != {img2.shape}.'
        return True

    def load_images(self, disaster_id, tile_id, tile):
        """Load images and mask from dataset paths"""
        data = {}
        data["pre_image"] = cv2.cvtColor(cv2.imread(tile["pre"]["image"]),
                                         cv2.COLOR_BGR2RGB)
        data["post_image"] = cv2.cvtColor(cv2.imread(tile["post"]["image"]),
                                          cv2.COLOR_BGR2RGB)
        data["pre_mask"] = cv2.imread(tile["pre"]["mask"])[:, :, 0]
        data["post_mask"] = cv2.imread(tile["post"]["mask"])[:, :, 0]

        self.same_shape(disaster_id, tile_id, data["pre_image"],
                        data["post_image"])
        self.same_shape(disaster_id, tile_id, data["post_image"],
                        data["pre_mask"])
        self.same_shape(disaster_id, tile_id, data["post_image"],
                        data["post_mask"])
        return data

    def __getitem__(self, i):
        disaster_id, tile_id, tile = self.tile_list[i]
        data = self.load_images(disaster_id, tile_id, tile)
        return disaster_id, tile_id, data


class RawDataset(TileDataset):
    """
        Class that inherits from TileDataset but it modifies its __getitem__
        method to return json files from the xBD raw dataset.
    """

    def __getitem__(self, i):
        disaster_id, tile_id, tile = self.tile_list[i]
        data = self.load_images(disaster_id, tile_id, tile)
        data["pre_json"] = read_json(tile["post"]["json"])
        data["post_json"] = read_json(tile["pre"]["json"])
        return disaster_id, tile_id, data
