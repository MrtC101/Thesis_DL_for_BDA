import os
import sys
from os.path import join
os.environ["PROJ_PATH"] = "/home/mrtc101/Desktop/tesina/repo/hiper_siames"
os.environ["SRC_PATH"] = join(os.environ["PROJ_PATH"], "src")
os.environ["DATA_PATH"] = join(os.environ["PROJ_PATH"], "data")
os.environ["OUT_PATH"] = join(os.environ["PROJ_PATH"], "out")
# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from train.train_pipeline import train_definitive
from utils.loggers.console_logger import LoggerSingleton

def train_definitive_model(configuration : dict) -> float:
    log = LoggerSingleton("Parameter Serach", 
                          folder_path=join(os.environ["OUT_PATH"], "hiper_console_logs"))
    # Train definitive model
    pred_path = os.path.join(os.environ['OUT_PATH'],"definitive_model")
    definitive_acc_score = train_definitive(pred_path, configuration)
    log.info(f"Accuracy for the final model : {definitive_acc_score}")    
    return definitive_acc_score