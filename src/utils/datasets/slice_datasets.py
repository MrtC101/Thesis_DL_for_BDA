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
        Dataset that uses SlicePathManager to read all sliced Dataset files.
    """
    pass

class SliceDataset(Dtaset):
    """
        Access Data using sliced_splits.json file.
    """
    pass