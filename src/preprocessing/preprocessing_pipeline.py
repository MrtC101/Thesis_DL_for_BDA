import os
import sys
from os.path import join

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.logger import LoggerSingleton

from preprocessing.prepare_folder.clean_folder import delete_not_in
from preprocessing.prepare_folder.create_label_masks import create_masks
from preprocessing.prepare_folder.delete_extra import leave_only_n
from preprocessing.raw.data_stdv_mean import create_data_dicts
from preprocessing.raw.split_raw_dataset import split_dataset
from preprocessing.sliced.make_smaller_tiles import slice_dataset
from preprocessing.sliced.split_sliced_dataset import split_sliced_dataset

def log_Title(title: str):
    """Prints a Title throw logger"""
    log = LoggerSingleton()
    log.info("="*50)
    log.info(f"{title.upper()}...")
    log.info("="*50)


def preprocess(disaster_num, border_width):
    """Pipeline sequence for data preprocessing."""
    
    # folder cleaning
    log_Title("deleting disasters that are not of interest")
    xbd_path = join(os.environ["DATA_PATH"], "xBD")
    raw_path = join(xbd_path, "raw")
    delete_not_in(raw_path)
    
    log_Title("creating target masks")
    create_masks(raw_path, border_width)
    
    log_Title("deleting extra disasters")
    leave_only_n(raw_path, disaster_num)
    
    # Raw data
    log_Title("split disasters")
    split_json_path = split_dataset(raw_path, xbd_path,{
        "train": 0.9,
        "test": 0.1
    })

    log_Title("creating data statistics")
    data_dicts_path = create_data_dicts(split_json_path, xbd_path)
    mean_std_json_path = os.path.join(data_dicts_path,"all_tiles_mean_stddev.json")
    
    ## ONLY FOR TRAIN DATASET
    #Imbalance treatment goes here

    #Data augmentation goes here

    ## 
        
    # Cropping
    log_Title("creating data patches")
    sliced_path = join(xbd_path, "sliced")
    slice_dataset(split_json_path, sliced_path)
    
    log_Title("split patches")
    split_sliced_json_path = split_sliced_dataset(sliced_path, split_json_path, xbd_path)
    
    return split_json_path, split_sliced_json_path, mean_std_json_path
