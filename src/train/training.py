# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from utils.datasets.shard_datasets import ShardDataset
from utils.visualization.raster_label_visualizer import RasterLabelVisualizer
from utils.common.files import read_json, dump_json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.siames.end_to_end_Siam_UNet import SiamUnet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from time import localtime, strftime
from datetime import datetime
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from train.phase import Phase
from utils.common.logger import get_logger
l = get_logger("training model")

logger_train : SummaryWriter = None
logger_val : SummaryWriter = None

def logging_wrapper(logger, phase):
    def decorator(func):
        def wrapper(*args):
            optimizer = args[1]['optimizer']
            epochs = args[1]['epochs']
            epoch = args[1]['epoch']

            if (phase == "train"):
                logger.add_scalar(
                    'learning_rate', optimizer.param_groups[0]["lr"], epoch)

            l.info(f'Model training for epoch {epoch}/{epochs}')
            start_time = datetime.now()

            result = func(*args)

            duration = datetime.now() - start_time
            logger.add_scalar(f'time_{phase}', duration.total_seconds(), epoch)

            return result
        return wrapper
    return decorator


@logging_wrapper(logger_train, "training")
def train(train_phase: Phase, epoch_context):
    """
    Train the model on dataset of the loader
    """
    confusion_mtrx_df_dmg, confusion_mtrx_df_bld, losses = train_phase.iteration(
        epoch_context)
    return confusion_mtrx_df_dmg, confusion_mtrx_df_bld


@logging_wrapper(logger_val, "validation")
def validation(val_phase: Phase, epoch_context):
    with torch.no_grad():
        confusion_mtrx_df_dmg, confusion_mtrx_df_bld, losses = val_phase.iteration(
            epoch_context)
    return confusion_mtrx_df_dmg, confusion_mtrx_df_bld, losses.avg


def resume_model(model : SiamUnet, training_config, starting_checkpoint_path):

    if starting_checkpoint_path and os.path.isfile(starting_checkpoint_path):
        l.info('Loading checkpoint from {}'.format(starting_checkpoint_path))
        optimizer, starting_epoch, best_acc = model.resume_from_checkpoint(training_config)
        l.info(
            f'Loaded checkpoint, starting epoch is {starting_epoch}, best f1 is {best_acc}')
    else:
        l.info('No valid checkpoint is provided. Start to train from scratch...')
        optimizer, starting_epoch, best_acc = model.resume_from_scratch(training_config)
    return optimizer, starting_epoch, best_acc


def output_directories(out_dir, exp_name):
    # set up directories (TrainPathManager?)
    exp_dir = os.path.join(out_dir, exp_name)

    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(logger_dir, exist_ok=True)

    evals_dir = os.path.join(exp_dir, 'evals')
    os.makedirs(evals_dir, exist_ok=True)

    config_dir = os.path.join(exp_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)

    return checkpoint_dir, logger_dir, evals_dir, config_dir


def train_model(train_config, path_config):

    # setup output directories
    checkpoint_dir, logger_dir, evals_dir, config_dir = \
        output_directories(path_config['out_dir'],path_config['exp_name'])
    dump_json(os.path.join(config_dir, 'train_config.txt'), train_config)
    dump_json(os.path.join(config_dir, 'path_config.txt'), path_config)

    global logger_train, logger_val
    logger_train = SummaryWriter(log_dir=logger_dir)
    logger_val = SummaryWriter(log_dir=logger_dir)

    # torch device
    l.info(f'Using PyTorch version {torch.__version__}.')
    device = torch.device(
        train_config['device'] if torch.cuda.is_available() else "cpu")
    l.info(f'Using device: {device}.')

    # DATA
    # Load datasets
    xBD_train = ShardDataset('train', path_config['shard_splits_json'])
    print('xBD_disaster_dataset train length: {}'.format(len(xBD_train)))
    xBD_val = ShardDataset('val', path_config['shard_splits_json'])
    print('xBD_disaster_dataset val length: {}'.format(len(xBD_val)))

    train_loader = DataLoader(xBD_train,
                              batch_size=train_config['batch_size'],
                              shuffle=True,
                              num_workers=8,
                              pin_memory=False)
    val_loader = DataLoader(xBD_val,
                            batch_size=train_config['batch_size'],
                            shuffle=False,
                            num_workers=8,
                            pin_memory=False)

    l.info('Get sample chips from train set...')
    sample_train_ids = xBD_train.get_sample_images(train_config['num_chips_to_viz'])
    l.info('Get sample chips from val set...')
    sample_val_ids = xBD_val.get_sample_images(train_config['num_chips_to_viz'])

    # TRAINING CONFIG

    # define model
    model = SiamUnet().to(device=device)
    l.info(model.model_summary())

    # resume from a checkpoint if provided
    optimizer, starting_epoch, best_acc = resume_model( model, train_config, path_config['starting_checkpoint_path'])

    # define loss functions and weights on classes
    global weights_loss, mode
    mode = train_config['mode']
    weights_seg_tf = torch.FloatTensor(train_config['weights_seg'])
    weights_damage_tf = torch.FloatTensor(train_config['weights_damage'])
    weights_loss = train_config['weights_loss']

    # loss functions
    criterion_seg_1 = \
        nn.CrossEntropyLoss(weight=weights_seg_tf).to(device=device)
    criterion_seg_2 = \
        nn.CrossEntropyLoss(weight=weights_seg_tf).to(device=device)
    criterion_damage = \
        nn.CrossEntropyLoss(weight=weights_damage_tf).to(device=device)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=2000, verbose=True)

    static_context = {
        'crit_seg_1': criterion_seg_1,
        'crit_seg_2': criterion_seg_2,
        'crit_dmg': criterion_damage,
        'device': device,
        "labels_set_dmg":  train_config['labels_dmg'],
        "labels_set_bld":  train_config['labels_bld'],
        "weights_loss": weights_loss,
        "label_map_json": path_config['label_map_json']
    }

    train_context = {
        'phase': "train",
        'logger': logger_train,
        'loader': train_loader,
        'sample_ids': sample_train_ids,
        'dataset': xBD_train
    }

    val_context = {
        'phase': "val",
        'logger': logger_val,
        'loader': val_loader,
        'sample_ids': sample_val_ids,
        'dataset':xBD_val
    }

    # Metrics
    cols = ['epoch', 'class', 'precision', 'recall', 'f1', 'accuracy']
    tr_dmg_metrics = pd.DataFrame(columns=cols)
    tr_bld_metrics = pd.DataFrame(columns=cols)
    val_dmg_metrics = pd.DataFrame(columns=cols)
    val_bld_metrics = pd.DataFrame(columns=cols)

    # epochs
    step_tr = 1
    epoch = starting_epoch
    epochs = train_config['epochs']

    epoch_context = {
        'epoch': epoch,
        'epochs': epochs,
        'step_tr': step_tr,
        'model': model,
        'optimizer': optimizer
    }

    # Objects for training
    train_phase = Phase(train_context, static_context)
    val_phase = Phase(val_context, static_context)

    while (epoch <= epochs):

        # train phase
        conf_mtrx_dmg_df_tr, conf_mtrx_bld_df_tr = train(train_phase, epoch_context)
        l.info(f'Compute actual metrics for model evaluation based on training set ...')
        tr_dmg_metrics, tr_bld_metrics, f1_harmonic_mean = \
            train_phase.compute_metrics(tr_bld_metrics, tr_dmg_metrics,
                                        conf_mtrx_dmg_df_tr, conf_mtrx_bld_df_tr,
                                        epoch_context)

        # val phase
        conf_mtrx_dmg_df_val, conf_mtrx_bld_df_val, losses_val = \
            validation(val_phase, epoch_context)
        scheduler.step(losses_val)  # decay Learning Rate
        l.info(f'Compute actual metrics for model evaluation based on validation set ...')
        val_bld_metrics, val_dmg_metrics, f1_harmonic_mean = \
            val_phase.compute_metrics(val_bld_metrics, val_dmg_metrics,
                                      conf_mtrx_dmg_df_val, conf_mtrx_bld_df_val,
                                      epoch_context)

        # compute average accuracy across all classes to select the best model
        val_acc_avg = f1_harmonic_mean
        is_best = val_acc_avg > best_acc
        best_acc = max(val_acc_avg, best_acc)

        l.info(
            f'Saved checkpoint for epoch {epoch}. Is it the highest f1 checkpoint so far: {is_best}\n')
        model.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_f1_avg': val_acc_avg,
            'best_f1': best_acc
        }, is_best, checkpoint_dir)

        epoch_context["epoch"] += 1

    logger_train.flush()
    logger_train.close()
    logger_val.flush()
    logger_val.close()
    l.info('Done')


if __name__ == "__main__":
    train_config = {
        'labels_dmg': [0, 1, 2, 3, 4],
        'labels_bld': [0, 1],
        'weights_seg': [1, 15],
        'weights_damage': [1, 35, 70, 150, 120],
        'weights_loss': [0, 0, 1],
        'mode': 'dmg',
        'init_learning_rate': 0.0005,  # dmg: 0.005, #UNet: 0.01,
        'device': 'cpu',
        'epochs': 1500,
        'batch_size': 32,
        'num_chips_to_viz': 1
    }
    path_config = {
        'exp_name': 'train_UNet',  # train_dmg
        'out_dir': '/home/mrtc101/Desktop/tesina/repo/my_siames/out',
        'shard_splits_json': '/home/mrtc101/Desktop/tesina/repo/my_siames/data/xBD/splits/shard_splits.json',
        'label_map_json': '/home/mrtc101/Desktop/tesina/repo/my_siames/data/constants/xBD_label_map.json',
        'starting_checkpoint_path': None
    }
    train_model(train_config, path_config)
