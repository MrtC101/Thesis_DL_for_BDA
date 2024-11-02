# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import multiprocessing
import pandas as pd
import torch
import random
from utils.common.pathManager import FilePath
from utils.loggers.console_logger import LoggerSingleton
from torch.utils.data import DataLoader


class TrainDataLoader(DataLoader):
    """
    Implementation of a DataLoader class with an extra method for image
    sampling.
    """
    last_num: int = 0
    last_sample: list
    seed: int = None

    def det_img_sample(self, number: int, normalized: bool) -> list:
        """
            Method that returns a random number of images from the dataloader
            and always are the same.

            Args:
                number : number of patches from TrainDataset to visualize.
                normalized : if patches are normalized or not.
        """
        if (number != self.last_num):
            # Establecer la semilla aleatoria
            if not self.seed:
                self.seed = random.randint(0, 2**31)

            random.seed(self.seed)

            # Seleccionar n índices aleatorios deterministas
            sample_idxs = random.sample(range(len(self)), number)

            # Obtener los elementos de los índices seleccionados
            self.dataset.set_normalize(normalized)

            self.last_sample = []
            for i in sample_idxs:
                self.last_sample.append(self.dataset[i])

            self.dataset.set_normalize(True)

        return self.last_sample


def build_fold_table(out_path: FilePath):
    config_folders = [fol_name for fol_name in out_path.get_folder_names()
                      if fol_name.startswith("config")]
    fold_list = []
    for dir in config_folders:
        conf_path = out_path.join(dir)
        for fold in conf_path.get_folder_names():

            fold_dir = conf_path.join(fold)
            metric_dir = fold_dir.join("metrics/csv")
            bld_df = pd.read_csv(metric_dir.join("val_bld_pixel_level.csv"))
            bld_df = bld_df.set_index(["epoch", "class"])
            dmg_df = pd.read_csv(metric_dir.join("val_dmg_pixel_level.csv"))
            dmg_df = dmg_df.set_index(["epoch", "class"])
            loss_df = pd.read_csv(metric_dir.join("val_loss.csv"))
            loss_df = loss_df.set_index(["epoch"])

            conf = dir[len(dir)-1]
            fold = fold[len(fold)-1]
            best_f1_hm = dmg_df["f1_harmonic_mean"].max()
            best_epoch = dmg_df["f1_harmonic_mean"].idxmax()[0]
            val_loss = loss_df.loc[best_epoch].iloc[0]
            seg_f1 = bld_df.loc[best_epoch, "f1_harmonic_mean"].iloc[0]
            row = [conf, fold, best_epoch, best_f1_hm, val_loss, seg_f1]
            fold_list.append(row)

    fold_df = pd.DataFrame(fold_list, columns=["Conf", "Fold", "Best epoch", "dmg-hf1", "val-loss", "seg-hf1"])
    fold_df = fold_df.sort_values(by=["Conf"])
    return fold_df


def get_best_config(out_path: FilePath, param_list: list) -> dict:
    best_conf = 0
    if (len(param_list) > 1):
        fold_df = build_fold_table(out_path)
        fold_df = fold_df.sort_values(by=["Conf", "Fold"])
        fold_df = fold_df.groupby('Conf').apply(lambda x: x.set_index('Fold'), include_groups=False)
        best_conf = fold_df["dmg-hf1"].idxmax()[0]
    return (best_conf, param_list[str(best_conf)])


def set_threads():
    """Configure PyTorch threads for performance."""
    physical_cores = multiprocessing.cpu_count()
    if torch.get_num_threads() < physical_cores or \
       torch.get_num_interop_threads() < physical_cores:
        log = LoggerSingleton()
        log.info(f'Using PyTorch version {torch.__version__}.')
        torch.set_num_threads(physical_cores)
        log.info(
            f"Number of threads for TorchScripts: {torch.get_num_threads()}")
        torch.set_num_interop_threads(physical_cores)
        log.info("Number of threads for PyTorch internal operations: " +
                 f"{torch.get_num_interop_threads()}")
