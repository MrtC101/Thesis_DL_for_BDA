from typing import overload


class RawDataset(Dataset):

    @overload
    def __init__(self, raw_path : str): 
        """
            Access raw data from xBD dataset using RawPathManager.
        """
        ...
        
    @overload
    def __init__(self, split_name: str, splits_json_path: str): 
        """
            Access raw data from xBD dataset using raw_splits.json file.
        """
        ...