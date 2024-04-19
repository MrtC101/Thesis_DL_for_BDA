
from utils.processed_data_path import load_processed_data
from .mask_creation import create_instance_mask
from .box_creation import create_bounding_boxes


def create_features(processed_data_path):
    # feature engineering
    disaster_zones_dict = load_processed_data(processed_data_path, "")
    create_instance_mask(disaster_zones_dict)
    create_bounding_boxes(disaster_zones_dict)
