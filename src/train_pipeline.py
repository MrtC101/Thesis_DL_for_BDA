import os
import sys

os.environ["SRC_PATH"] = "/home/mrtc101/Desktop/tesina/repo/my_siames/src"
os.environ["DATA_PATH"] = "/home/mrtc101/Desktop/tesina/repo/my_siames/data"

if(os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.visualization.logger import get_logger
l = get_logger("delete_extra")


