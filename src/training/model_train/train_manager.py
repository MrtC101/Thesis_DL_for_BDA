# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import torch
import pandas as pd
from typing import Tuple
from tqdm import trange
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from models.trainable_model import TrainModel

from training.model_train.epoch_manager import EpochManager
from training.model_train.utils import TrainDataLoader

from utils.common.pathManager import FilePath
from utils.metrics.loss_manager import LossManager
from utils.metrics.metric_manager import MetricManager
from utils.loggers.console_logger import LoggerSingleton
from utils.loggers.tensorboard_logger import TensorBoardLogger

log = LoggerSingleton()


def save_configs(config_dir: FilePath, configs: dict):
    """
    Save configuration parameters to JSON and LaTeX files.

    Args:
        config_dir (FilePath): Directory path for saving configuration files.
        configs (dict): Configuration parameters.
    """
    config_dir.join('configs.json').save_json(configs)
    df = pd.DataFrame(data=[list(configs.keys()), list(configs.values())]).transpose()
    df.columns = ["Parameters", "Values"]
    df.to_latex(config_dir.join('configs.tex'))


def save_if_best(metrics_df: pd.DataFrame, best_hf1: float, checkpoint_dir: FilePath,
                 model: TrainModel, optimizer: Optimizer, epoch: int,
                 save_all_checkpoints: bool) -> float:
    """
    Save the checkpoint if the current model has the best f1_harmonic_mean
    score.

    Args:
        metrics_df: DataFrame of metrics.
        best_acc: Best f1_harmonic_mean score so far.
        checkpoint_dir: Directory to save the checkpoint.
        model: Model to save.
        optimizer: Optimizer for the model.
        epoch: Current epoch number.

    Returns:save_all_checkpoitns
        float: Updated best accuracy score.
    """
    pixel_hf1 = metrics_df["dmg_pixel_level"]["f1_harmonic_mean"].mean()
    is_best = pixel_hf1 >= best_hf1
    best_score = max(pixel_hf1, best_hf1)
    if save_all_checkpoints or is_best:
        log.info(f'Saved checkpoint for epoch {epoch}. Highest f1 checkpoint so far: {is_best}\n')

        model.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_f1_avg': pixel_hf1,
            'best_f1': best_score
        }, is_best, checkpoint_dir)
    return best_score


def resume_model(model: TrainModel, checkpoint_path: FilePath, checkpoint: bool,
                 device, init_learning_rate, tb_logger, freeze_seg):
    """
    Resume the model from a checkpoint or start from scratch.

    Args:
        model: Model to resume or initialize.
        checkpoint_path: Path to the checkpoint.
        device: Device for model training.
        init_learning_rate: Initial learning rate.
        tb_logger: TensorBoard logger.
        new_optimizer: Boolean to determine if segmentation branch is freezed.

    Returns:
        tuple: (optimizer, starting_epoch, best_acc)
    """

    files = len(checkpoint_path.get_files_names())
    if checkpoint and files > 0:
        log.info(f'Loading checkpoint from {checkpoint_path} ...')
        return model.resume_from_checkpoint(checkpoint_path, device, init_learning_rate,
                                            tb_logger, freeze_seg)
    else:
        log.info('No valid checkpoint provided. Training from scratch...')
        return model.resume_from_scratch(init_learning_rate)


def get_dirs(out_dir: FilePath) -> Tuple[FilePath]:
    """
    Create directories for the current experiment.

    Args:
        out_dir: Path for the main output folder.

    Returns:
        tuple: (checkpoint_dir, tb_logger_dir, config_dir, metric_dir)
    """
    out_dir.create_folder()
    return (
        out_dir.join('checkpoints').create_folder(),
        out_dir.join('tb_logs').create_folder(),
        out_dir.join('configs').create_folder(),
        out_dir.join('metrics').create_folder()
    )


def train_model(configs: dict[str], paths: dict[str], train_loader: TrainDataLoader,
                val_loader: TrainDataLoader, save_all_checkpoints: bool = True):
    """
    Train the model using the specified configurations.

    Args:
        configs (dict): Configuration parameters.
        paths (dict): Paths for saving files and logs.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.

    Returns:
        float: The best accuracy score achieved.
    """

    out_dir = FilePath(paths['out_dir'])
    log = LoggerSingleton(out_dir.basename().capitalize(), folder_path=out_dir)

    # setup output directories
    checkpoint_dir, tb_logger_dir, config_dir, metric_dir = get_dirs(out_dir)
    save_configs(config_dir, configs)

    # Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f'Using device: {device}.')
    model = TrainModel().to(device)

    # samples are for tensorboard visualization of same images through epochs
    tb_logger = TensorBoardLogger(tb_logger_dir, configs['num_chips_to_viz'])

    # resume from a checkpoint if provided
    optimizer, starting_epoch, best_score = resume_model(model, checkpoint_dir,
                                                         configs["checkpoint"],
                                                         device, configs['learning_rate'],
                                                         tb_logger, configs['freeze_seg'])
    log.info(f"Loaded checkpoint, starting epoch is {starting_epoch}, best f1 is {best_score}")

    # loss functions
    w_seg = torch.FloatTensor(configs['weights_seg'])
    w_damage = torch.FloatTensor(configs['weights_dmg'])
    criterion_seg = nn.CrossEntropyLoss(weight=w_seg).to(device=device)
    criterion_damage = nn.CrossEntropyLoss(weight=w_damage).to(device=device)
    criterions = [criterion_seg, criterion_seg, criterion_damage]

    # managers
    loss_manager = LossManager(configs['weights_loss'], criterions)
    metric_manager = MetricManager(configs['labels_bld'], configs['labels_dmg'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=configs["RLROP_patience"], threshold=0.0001)

    epochs = configs['tot_epochs']

    sheared_vars = {
        'tb_logger': tb_logger,
        'tot_epochs': epochs,
        'optimizer': optimizer,
        'model': model,
        'device': device,
    }

    # Objects for training
    training = EpochManager(mode=EpochManager.Mode.TRAINING,
                            loader=train_loader,
                            loss_manager=LossManager(configs['weights_loss'], criterions),
                            metric_manager=MetricManager(
                                configs['labels_bld'], configs['labels_dmg']),
                            **sheared_vars)
    validation = EpochManager(mode=EpochManager.Mode.VALIDATION, loader=val_loader,
                              loss_manager=LossManager(configs['weights_loss'], criterions),
                              metric_manager=MetricManager(
                                  configs['labels_bld'], configs['labels_dmg']),
                              **sheared_vars)

    # Metrics
    train_metrics = []
    val_metrics = []
    train_loss = []
    val_loss = []

    for epoch in trange(starting_epoch, epochs+1, desc="Epoch"):
        # TRAINING
        train_epoch_metrics, tr_epoch_loss = training.run_epoch(epoch)
        train_metrics.append(train_epoch_metrics)
        train_loss.append({"epoch": epoch, "loss": tr_epoch_loss})
        # VALIDATION
        with torch.no_grad():
            val_epoch_metrics, val_epoch_loss = validation.run_epoch(epoch)
        val_metrics.append(val_epoch_metrics)
        val_loss.append({"epoch": epoch, "loss": val_epoch_loss})

        scheduler.step(val_epoch_loss)  # decay Learning Rate

        log.info(f"epoch {epoch}/{configs['tot_epochs']}:" +
                 f"train loss:{tr_epoch_loss:3f};" +
                 f"val loss:{val_epoch_loss:3f};")
        # CHECKPOINT
        best_score = save_if_best(val_epoch_metrics, best_score, checkpoint_dir, model,
                                  optimizer, epoch, save_all_checkpoints)

    MetricManager.save_metrics(train_metrics, train_loss, metric_dir, "train")
    MetricManager.save_metrics(val_metrics, val_loss, metric_dir, "val")

    tb_logger.flush()
    tb_logger.close()
