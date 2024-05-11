import os
import sys

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.files import dump_json, is_dir
from utils.common.logger import get_logger
from utils.pathManagers.shardManager import ShardPathManager
l = get_logger("Compute data from images")

def split_shard_dataset(shards_path : str, out_path):

    split_path = os.path.join(out_path,"splits")
    os.makedirs(split_path,exist_ok=True)

    shards_split_dict = ShardPathManager().load_paths(shards_path)

    split_json_path = os.path.join(split_path,"shard_splits.json")
    dump_json(split_json_path,shards_split_dict)
    return split_json_path