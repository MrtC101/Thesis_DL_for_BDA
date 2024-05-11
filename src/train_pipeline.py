import os
import sys

os.environ["SRC_PATH"] = "/home/mrtc101/Desktop/tesina/repo/my_siames/src"
os.environ["DATA_PATH"] = "/home/mrtc101/Desktop/tesina/repo/my_siames/data"

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from preprocessing.sliced.split_sliced_dataset import split_sliced_dataset
from utils.common.logger import get_logger
l = get_logger("Compute data from images")

from preprocessing.prepare_folder.clean_folder import delete_not_in
from preprocessing.prepare_folder.delete_extra import leave_only_n
from preprocessing.prepare_folder.create_label_masks import create_masks
from preprocessing.raw.data_stdv_mean import create_data_dicts
from preprocessing.raw.split_raw_dataset import split_dataset
from preprocessing.sliced.make_smaller_tiles import slice_dataset
from preprocessing.sliced.split_sliced_dataset import split_sliced_dataset
from preprocessing.shards.make_data_shards import create_shards
from preprocessing.shards.split_shard_dataset import split_shard_dataset

xbd_path = os.path.join(os.environ["DATA_PATH"],"xBD")
raw_path = os.path.join(xbd_path,"raw")
# folder cleaning
#delete_not_in(raw_path)
#create_masks(raw_path,1)
#leave_only_n(raw_path,40)
# split json
split_json_path = split_dataset(raw_path,xbd_path)
data_dicts_path = create_data_dicts(split_json_path,xbd_path)
# Sliced
sliced_path = os.path.join(xbd_path,"sliced")
slice_dataset(split_json_path,sliced_path)
split_sliced_json_path = split_sliced_dataset(sliced_path,split_json_path,xbd_path)
# Sharded
mean_stddev_json = os.path.join(data_dicts_path,"all_tiles_mean_stdev.json")
shards_path = os.path.join(xbd_path,"shards")
create_shards(split_sliced_json_path,mean_stddev_json,shards_path,4)
split_shard_json_path = split_shard_dataset(shards_path,xbd_path)

# train
# evaluate with test
# inference