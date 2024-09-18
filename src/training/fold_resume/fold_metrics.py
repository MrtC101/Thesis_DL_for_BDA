import pandas as pd
from utils.common.pathManager import FilePath

def build_fold_table(out_path : FilePath):
    config_folders = [fol_name for fol_name in out_path.get_folder_names() if fol_name.startswith("config")]
    fold_list = []
    for dir in config_folders:
        conf_path = out_path.join(dir)
        for fold in conf_path.get_folder_names():
            
            fold_dir = conf_path.join(fold)
            metric_dir = fold_dir.join("metrics/csv")
            bld_df = pd.read_csv(metric_dir.join("val_bld_pixel_level.csv"))
            bld_df = bld_df.set_index(["epoch","class"])
            dmg_df = pd.read_csv(metric_dir.join("val_dmg_pixel_level.csv"))
            dmg_df = dmg_df.set_index(["epoch","class"])
            loss_df = pd.read_csv(metric_dir.join("val_loss.csv"))
            loss_df = loss_df.set_index(["epoch"])
            
            c = dir[len(dir)-1]
            f = fold[len(fold)-1]
            best_f1_hm = dmg_df["f1_harmonic_mean"].max()
            best_epoch = dmg_df["f1_harmonic_mean"].idxmax()[0]
            val_loss = loss_df.loc[best_epoch].iloc[0]
            seg_f1 = bld_df.loc[best_epoch,"f1"].iloc[0]
            row = [c, f, best_epoch, best_f1_hm, val_loss, seg_f1]
            fold_list.append(row)

    fold_df = pd.DataFrame(fold_list, columns=["Conf", "Fold", "Best epoch", "val-loss", "Harmonic-mean-f1", "Seg-f1"])
    fold_df = fold_df.sort_values(by=["Conf"])
    return fold_df

def get_best_config(out_path: FilePath, param_list: list) -> dict:
    fold_df = build_fold_table(out_path)
    fold_df = fold_df.sort_values(by=["Conf","Fold"])
    fold_df = fold_df.groupby('Conf').apply(lambda x: x.set_index('Fold'), include_groups=False)
    best_conf = fold_df["Harmonic-mean-f1"].idxmax()[0]
    return param_list[best_conf]
