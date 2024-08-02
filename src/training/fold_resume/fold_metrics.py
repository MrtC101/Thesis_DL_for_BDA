from collections import defaultdict
import pandas as pd

from utils.common.pathManager import FilePath


def compute_config_resume(config_path: FilePath) -> dict:
    best_fold = {"fold_num": -1, "epoch": -1, "val_loss": 10.0}

    for file_name in config_path.get_folder_names():
        fold_folder = config_path.join(file_name)
        metric_dir = fold_folder.join("metrics","csv")

        loss_df = pd.read_csv(metric_dir.join("val_loss.csv"), index_col="epoch")
        best_epoch = loss_df["loss"].idxmin()
        fold_resume = []

        for file in metric_dir.get_files_names():
            if not file.find("loss") > -1 and file.endswith(".csv"):
                split, mask, level, _ = file.split("_")
                df = pd.read_csv(metric_dir.join(file), index_col="epoch")
                df["split"] = split
                df["mask"] = mask
                res = df.loc[best_epoch]

                if isinstance(res, pd.Series):
                    res = res.to_frame().transpose()

                res = res.reset_index().rename(columns={"index": "epoch"})
                fold_resume.append(res)

        resume_df = pd.concat(fold_resume, axis=0, ignore_index=True)
        resume_df.to_csv(fold_folder.join("resume.csv"), index=False)

        fold_num = int(file_name.split("_")[-1])
        curr_fold = {
            "fold_num": fold_num,
            "epoch": best_epoch,
            "val_loss": loss_df.loc[best_epoch].iloc[0]
        }
        best_fold = min(best_fold, curr_fold, key=lambda x: x["val_loss"])
    return best_fold


def get_best_config(out_path: FilePath, param_list: list) -> dict:
    results = []
    for folder_name in out_path.get_folder_names():
        if folder_name.startswith("config"):
            config_out_path = out_path.join(folder_name)
            result = compute_config_resume(config_out_path)
            result["conf_num"] = folder_name[len(folder_name)-1]
            results.append(result)
    res_df = pd.DataFrame(results)
    res_df.set_index("conf_num")
    best_idx = res_df["val_loss"].idxmax()
    return param_list[best_idx]
