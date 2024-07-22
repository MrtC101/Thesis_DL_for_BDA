import pandas as pd

from utils.common.pathManager import FilePath

def compute_config_resume(config_path : FilePath) -> dict:
    for fold_folder in config_path.get_folder_paths():
        metric_dir = fold_folder.join("metric")
        
    return

def get_best_config(out_path : FilePath, param_list : dict) -> dict:
    results = []
    for folder_name in out_path.get_folder_names():
        if folder_name.startswith("config"):
            config_out_path = out_path.join(folder_name)
            result = compute_config_resume(config_out_path)
            results.append(result)
    res_df = pd.DataFrame(results)
    best_idx = res_df.min()
    return param_list[best_idx]