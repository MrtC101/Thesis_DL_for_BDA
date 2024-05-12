import os
import sys

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from os.path import join
from utils.common.files import is_dir, is_npy
from utils.common.defaultDictFactory import nested_defaultdict

class ShardPathManager:

    def _add_original_images(self,subset,patch_dict,patch,split_dict):
        dis_id,tile_id,patch_id = patch.split("_")
        for time in ["pre","post"]:
            img_path = split_dict[subset][dis_id][tile_id][time]["image"]
            patch_dict[subset][dis_id][tile_id][patch_id][f"org_{time}"] = img_path
    
    def _add_patch_files(self,subset,sliced_dict,file,file_path):
        file_name : str = file.split(".")[0]
        dis_id,tile_id,patch_id,type = file_name.split("_")
        sliced_dict[subset][dis_id][tile_id][patch_id][type] = file_path
    
    def load_paths(self,shard_dir_path: str) -> dict:
        """
            Creates a DisasterDict that stores each file path
        """
        is_dir(shard_dir_path) 
        shards_dict = nested_defaultdict(5,str)
        for subset in ["train","val"]:
            subset_path = join(shard_dir_path,subset)
            dataset_shards = sorted(os.listdir(subset_path))
            for shard in dataset_shards:
                # assert para los datos
                shard_path = join(subset_path,shard)
                is_npy(shard_path)
                shard_name = shard.split(".")[0]
                split_id, type_id, shard_id = shard_name.split("_")
                shards_dict[split_id][type_id][shard_id] = shard_path
        return shards_dict
    
    def load_dataset(disaster_splits_json,disaster_mean_stddev,shards_dir,shard_num):
        splits = read_json(disaster_splits_json)
        data_mean_stddev = read_json(disaster_mean_stddev)

        train_ls = [] 
        val_ls = []
        for item, val in splits.items():
            train_ls += val['train'] 
            val_ls += val['val']
        xBD_train = ShardDataset(shards_dir, shard_num, 'train', data_mean_stddev, transform=True, normalize=True)
        xBD_val = DisasterDataset(shards_dir, shard_num, 'val', data_mean_stddev, transform=False, normalize=True)

        print('xBD_disaster_dataset train length: {}'.format(len(xBD_train)))
        print('xBD_disaster_dataset val length: {}'.format(len(xBD_val)))

        return xBD_train, xBD_val