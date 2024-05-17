# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from utils.datasets.shard_datasets import ShardDataset
from utils.common.files import is_dir
from models.siames.end_to_end_Siam_UNet import SiamUnet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os
import sys
from utils.common.logger import LoggerSingleton

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
log = LoggerSingleton()


def resume_model(l, model: SiamUnet, training_config,
                 starting_checkpoint_path):
    if starting_checkpoint_path and os.path.isfile(starting_checkpoint_path):
        log.info('Loading checkpoint from {}'.format(starting_checkpoint_path))
        _, last_epoch, _ = model.resume_from_checkpoint(training_config)
        log.info(f"loaded checkpoint at {starting_checkpoint_path}.")
    else:
        log.info('No valid checkpoint is provided.')
    return last_epoch


def output_directories(out_dir, exp_name):
    # set up directories (TrainPathManager?)
    is_dir(out_dir)
    exp_dir = os.path.join(out_dir, exp_name)

    # set up directories
    c_logger_dir = os.path.join(output_dir, exp_name, 'test_logs')
    os.makedirs(c_logger_dir, exist_ok=True)

    tb_logger_dir = os.path.join(output_dir, exp_name, 'tb_test_logs')
    os.makedirs(tb_logger_dir, exist_ok=True)

    evals_dir = os.path.join(output_dir, exp_name, 'evals')
    os.makedirs(evals_dir, exist_ok=True)

    output_dir = os.path.join(output_dir, exp_name, 'output')
    os.makedirs(output_dir, exist_ok=True)

    return c_logger_dir, tb_logger_dir, evals_dir, output_dir


def test_model(test_config, path_config):

    # setup output directories
    tb_logger_dir, evals_dir = output_directories(
        path_config['out_dir'], path_config['exp_name'])
    logger_test = SummaryWriter(log_dir=tb_logger_dir)
    log.name = "testing model"

    # torch device
    log.info(f'Using PyTorch version {torch.__version__}.')
    device = torch.device(
        test_config['device'] if torch.cuda.is_available() else "cpu")
    log.info(f'Using device: {device}.')

    # DATA
    # Load datasets
    xBD_test = ShardDataset('test', path_config['shard_splits_json'])
    log.info('xBD_disaster_dataset test length: {}'.format(len(xBD_test)))

    test_loader = DataLoader(xBD_test,
                             batch_size=test_config['batch_size'],
                             shuffle=True,
                             num_workers=8,
                             pin_memory=False)

    log.info('Get sample chips from test set...')
    sample_test_ids = xBD_test.get_sample_images(
        test_config['num_chips_to_viz'])

    # TRAINING CONFIG

    # define model
    model = SiamUnet().to(device=device)
    log.info(model.model_summary())

    # resume from a checkpoint if provided
    epoch = resume_model(l, model, test_config,
                         path_config['starting_checkpoint_path'])

    # define loss functions and weights on classes
    weights_seg_tf = torch.FloatTensor(test_config['weights_seg'])
    weights_damage_tf = torch.FloatTensor(test_config['weights_damage'])
    weights_loss = test_config['weights_loss']

    # loss functions
    criterion_seg_1 = \
        nn.CrossEntropyLoss(weight=weights_seg_tf).to(device=device)
    criterion_seg_2 = \
        nn.CrossEntropyLoss(weight=weights_seg_tf).to(device=device)
    criterion_damage = \
        nn.CrossEntropyLoss(weight=weights_damage_tf).to(device=device)

    static_context = {
        'crit_seg_1': criterion_seg_1,
        'crit_seg_2': criterion_seg_2,
        'crit_dmg': criterion_damage,
        'device': device,
        "labels_set_dmg":  test_config['labels_dmg'],
        "labels_set_bld":  test_config['labels_bld'],
        "weights_loss": weights_loss,
        "label_map_json": path_config['label_map_json']
    }

    test_context = {
        'phase': "test",
        'logger': logger_test,
        'loader': test_loader,
        'sample_ids': sample_test_ids,
        'dataset': xBD_test
    }

    # Objects for testing
    test = Phase(test_context, static_context)

    # TEST STEP

    with torch.no_grad():
        conf_mtrx_dmg_df_test, conf_mtrx_bld_df_test, _, \
            conf_mtrx_df_dmg_bld_level_test = test.epoch_iteration(
                model)

    log.info(f'Compute actual metrics for model \
             evaluation based on test set ...')

    # damage level eval validation (pixelwise)
    dmg_metrics, f1_harmonic_mean = test.dmg_metric.compute_eval_metrics(
        epoch, conf_mtrx_dmg_df_test)
    dmg_metrics.loc[len(dmg_metrics.index)] = {
        'class': 'harmonic-mean-of-all', 'precision': '-', 'recall': '-',
        'f1': f1_harmonic_mean, 'accuracy': '-'}

    # bld detection eval validation (pixelwise)
    bld_metrics, _ = test.bld_metric.compute_eval_metrics(
        epoch, conf_mtrx_bld_df_test)

    # damage level eval validation (building-level)
    dmg_bld_metrics, f1_harmonic_mean = test.dmg_bld_metric.compute_eval_metrics(
        epoch, conf_mtrx_df_dmg_bld_level_test)
    dmg_bld_metrics.loc[len(dmg_bld_metrics.index)] = {
        'class': 'harmonic-mean-of-all', 'precision': '-', 'recall': '-',
        'f1': f1_harmonic_mean, 'accuracy': '-'}

    # save confusion metrices
    conf_mtrx_bld_df_test.to_csv(os.path.join(
        evals_dir, 'confusion_mtrx_bld.csv'), index=False)
    conf_mtrx_dmg_df_test.to_csv(os.path.join(
        evals_dir, 'confusion_mtrx_dmg.csv'), index=False)
    conf_mtrx_df_dmg_bld_level_test.to_csv(os.path.join(
        evals_dir, 'confusion_mtrx_dmg_building_level.csv'), index=False)

    # save evalution metrics
    bld_metrics.to_csv(os.path.join(
        evals_dir, 'eval_results_bld.csv'), index=False)
    dmg_metrics.to_csv(os.path.join(
        evals_dir, 'eval_results_dmg.csv'), index=False)
    dmg_bld_metrics.to_csv(os.path.join(
        evals_dir, 'eval_results_dmg_building_level.csv'), index=False)

    logger_test.flush()
    logger_test.close()
    log.info('Done')


if __name__ == "__main__":
    test_config = {
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
    test_model(test_config, path_config)
