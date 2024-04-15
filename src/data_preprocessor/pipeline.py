from .utils.path_manager import load_raw_data, organize_dataset, load_processed_data, folder_to_dataframe
from .utils.path_manager import DisasterZone
from .box_creation import create_bounding_boxes
from .mask_creation import create_instance_mask
from .data_balancer import data_augment, sampling
import pandas as pd
import os

def feature_engineering(raw_data_path, processed_data_path):
    # feature engineering
    disaster_zones_dict: dict[DisasterZone] = load_raw_data(raw_data_path)
    create_instance_mask(disaster_zones_dict,
                         raw_data_path, processed_data_path)
    create_bounding_boxes(disaster_zones_dict,
                          raw_data_path, processed_data_path)
    organize_dataset(disaster_zones_dict, raw_data_path, processed_data_path)

def balance_data(raw_data_path, processed_data_path):
    # balance data
    folder_dict = load_processed_data(processed_data_path)
    zone_df = folder_to_dataframe(folder_dict)
    data_augment(folder_dict,zone_df)
    
    aug_folder_list = load_processed_data(processed_data_path)
    aug_zone_df = folder_to_dataframe(aug_folder_list)
    balance_df = sampling(aug_zone_df)

    return (aug_folder_list,balance_df)
    

def preprocess_pipeline(data_path) -> pd.DataFrame:
    """
        Data pre-process pipeline
    """
    raw_data_path = data_path + "/raw_dataset"
    processed_data_path = data_path + "/processed_dataset"
    if (os.path.exists(data_path+"/train")):
        os.rename(data_path+"/train", raw_data_path)

    if(not os.path.exists(processed_data_path)):
        feature_engineering(raw_data_path, processed_data_path)

    return balance_data(raw_data_path, processed_data_path)


if __name__ == "__main__":
    pass
