# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys
from utils.common.files import dump_json
from utils.pathManagers.shardManager import ShardPathManager
from utils.common.logger import LoggerSingleton

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
log = LoggerSingleton()


def split_shard_dataset(shards_path: str, out_path: str) -> str:
    """
    Args:
        shards_path: Path to the `Shard` folder where are stored all the shard
        files.
        out_path: Path to the folder where is `splits` folder or will be
        created.
    Return:
        str: Path to the new `shard_split` JSON file.
    Example:
        >>> split_shard_dataset("data/xBD/shards","data/xBD")
    """
    log.name="Split Shards"
    split_path = os.path.join(out_path, "splits")
    os.makedirs(split_path, exist_ok=True)

    shards_split_dict = ShardPathManager().load_paths(shards_path)

    split_json_path = os.path.join(split_path, "shard_splits.json")
    dump_json(split_json_path, shards_split_dict)
    return split_json_path
