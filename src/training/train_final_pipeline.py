# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
"""Train Pipeline

This module establishes the sequence of execution of
scripts inside `preprocessing`, `train` and `test`
packages, to implement the data preprocessing,
model training and testing pipeline for this project.
"""
import torch
import multiprocessing
from training.model_train.train_manager import train_model
from training.model_train.run_on_test_pipeline import inference_on_test
from training.model_train.utils import set_threads
from utils.common.pathManager import FilePath
from utils.common.timeManager import measure_time
from utils.dataloaders.train_dataloader import TrainDataLoader
from utils.loggers.console_logger import LoggerSingleton
from utils.datasets.train_dataset import TrainDataset


@measure_time
def train_final_model(configs: dict[str, any], paths: dict[str, any]):
    """
    Perform a training of the model.

    Args:
        configs (dict): Configuration parameters.
        paths (dict): Paths used in the process.

    Returns:
        float: Mean accuracy score over all folds.
    """
    # Logger
    out_dir = FilePath(paths['out_dir'])
    log_out = out_dir.join("console_logs")
    log = LoggerSingleton("FINAL MODEL", folder_path=log_out)
    log.info(f"Using best configuration with number {configs[0]}")
    configs = configs[1]

    set_threads()

    configs['tot_epochs'] = configs['final_epochs']

    # Load Dataset
    xBD_train = TrainDataset('train', paths['split_json'], paths['mean_json'])
    train_loader = TrainDataLoader(xBD_train, batch_size=configs['batch_size'])
    log.info(f'xBD_disaster_dataset TRAIN length: {len(xBD_train)}')

    xBD_val = TrainDataset('val', paths['split_json'], paths['mean_json'])
    val_loader = TrainDataLoader(xBD_val, batch_size=configs['batch_size'])
    log.info(f'xBD_disaster_dataset VAL length: {len(xBD_train)}')

    train_model(configs, paths, train_loader, val_loader)

    xBD_test = TrainDataset('test', paths['split_json'], paths['mean_json'])
    test_loader = TrainDataLoader(xBD_test, batch_size=configs['batch_size'])
    log.info(f'xBD_disaster_dataset TEST length: {len(xBD_test)}')
    inference_on_test(configs, paths, test_loader)
