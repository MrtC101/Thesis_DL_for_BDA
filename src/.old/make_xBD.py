import pandas as pd
import os
from utils.raw_data_path import organize_dataset
from .features_engineering.build_features import create_features
from .data_processor.data_balancer import balance_data


def preprocess_data(data_path) -> pd.DataFrame:
    """
        Data pre-process pipeline
    """
    raw_data_path = os.path.join(data_path, "raw", "train")
    processed_data_path = os.path.join(data_path, "processed")

    if (os.path.exists(raw_data_path)):
        organize_dataset(raw_data_path, processed_data_path)

    create_features(processed_data_path)

    augmented_data_path = os.path.join(data_path, "augmented")
    if (not os.path.exists(augmented_data_path)):
        os.mkdir(augmented_data_path)
    return balance_data(processed_data_path, augmented_data_path)
