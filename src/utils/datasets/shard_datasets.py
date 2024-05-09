import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.logger import get_logger
l = get_logger("delete_extra")

import cv2
from torch.utils.data import Dataset
from utils.pathManagers.rawManager import RawPathManager
from utils.common.files import read_json, is_dir, is_json

class PatchDataset(Dataset):
    """
        Dataset that uses ShardPathManager to read all sharded Dataset files.
    """
    pass

class ShardDataset(Dataset):
    """
        Access Data using shard_splits.json file.
    """
    pass


class oldDataset(Dataset):

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
