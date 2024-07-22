# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
"""Train Pipeline

This module establishes the sequence of execution of
scripts inside `preprocessing`, `train` and `test`
packages, to implement the data preprocessing,
model training and testing pipeline for this project.
"""
import torch
from training.model_train.train_manager import train_model
from utils.common.pathManager import FilePath
from utils.dataloaders.train_dataloader import TrainDataLoader
from utils.loggers.console_logger import LoggerSingleton
from utils.datasets.train_dataset import TrainDataset


def start_train(configs: dict, paths: dict, xBD_train, xBD_test=None,
                train_sampler=None, val_sampler=None):
    """
    Start the training process based on the provided configurations
      and dataset splits.

    Args:
        configs (dict): Configuration dictionary.
        paths (dict): Dictionary of paths used in the process.
        xBD_train: Training dataset.
        xBD_test: Optional test dataset.
        train_idx: Optional training indices for cross-validation.
        val_idx: Optional validation indices for cross-validation.

    Returns:
        float: The score of the training process.
    """
    # Create the data loaders
    train_loader = TrainDataLoader(
        xBD_train,
        batch_size=configs['batch_size'],
        num_workers=configs['batch_workers'],
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler
    )
    val_loader = TrainDataLoader(
        xBD_train,
        batch_size=configs['batch_size'],
        num_workers=configs['batch_workers'],
        pin_memory=False,
        shuffle=(val_sampler is None),
        sampler=val_sampler
    )
    test_loader = TrainDataLoader(
        xBD_test,
        batch_size=configs['batch_size'],
        shuffle=False,
        num_workers=configs['batch_workers'],
        pin_memory=False
    ) if xBD_test else None
    score = train_model(configs, paths, train_loader, val_loader,
                        test_loader=test_loader)

    return score


def set_threads(torch_threads, torch_op_threads):
    """
    Configure PyTorch threads for performance.

    Args:
        torch_threads: Number of threads for PyTorch.
        torch_op_threads: Number of inter-op threads for PyTorch.
    """
    if (torch.get_num_threads() < torch_threads or
       torch.get_num_interop_threads() < torch_op_threads):
        log = LoggerSingleton()
        log.info(f'Using PyTorch version {torch.__version__}.')
        torch.set_num_threads(torch_threads)
        log.info(
            f"Number of threads for TorchScripts: {torch.get_num_threads()}")
        torch.set_num_interop_threads(torch_op_threads)
        log.info("Number of threads for PyTorch internal operations: " +
                 f"{torch.get_num_interop_threads()}")


def train_definitive(configs: dict[str, any], paths: dict[str, any]):
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
    log = LoggerSingleton("DEFINITIVE MODEL", folder_path=log_out)

    set_threads(configs['torch_threads'], configs['torch_op_threads'])

    # Load Dataset
    xBD_train = TrainDataset('train', paths['split_json'], paths['mean_json'])
    log.info(f'xBD_disaster_dataset train length: {len(xBD_train)}')
    xBD_test = TrainDataset('test', paths['split_json'], paths['mean_json'])
    log.info(f'xBD_disaster_dataset test length: {len(xBD_test)}')

    return start_train(configs, paths, xBD_train, xBD_test)