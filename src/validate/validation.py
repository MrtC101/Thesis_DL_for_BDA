# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.siames.end_to_end_Siam_UNet import SiamUnet
from utils.common.files import dump_json, is_dir
from utils.datasets.shard_datasets import ShardDataset
from train.phase import Phase
import os
import sys
from utils.common.logger import LoggerSingleton
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
log = LoggerSingleton()


def resume_model(model: SiamUnet, checkpoint_path, tb_log_dir, training_config):
    """Choose how to load model parameters"""
    if checkpoint_path and os.path.isfile(checkpoint_path):
        log.info('Loading checkpoint from {}'.format(checkpoint_path))
        _, last_epoch, _ = model.resume_from_checkpoint(checkpoint_path,
                                                        tb_log_dir,
                                                        training_config)
        log.info(f"loaded checkpoint at {checkpoint_path}.")
    else:
        log.info('No valid checkpoint is provided.')
        raise Exception("No weights for model to load.")
    return last_epoch


def output_directories(out_dir, exp_name):
    """Creates out output folders"""
    # set up directories (TrainPathManager?)
    is_dir(out_dir)
    exp_dir = os.path.join(out_dir, exp_name)

    # set up directories
    c_logger_dir = os.path.join(exp_dir, 'test_logs')
    os.makedirs(c_logger_dir, exist_ok=True)

    tb_logger_dir = os.path.join(exp_dir, 'tb_test_logs')
    os.makedirs(tb_logger_dir, exist_ok=True)

    evals_dir = os.path.join(exp_dir, 'evals')
    os.makedirs(evals_dir, exist_ok=True)

    output_dir = os.path.join(exp_dir, 'predictions')
    os.makedirs(output_dir, exist_ok=True)

    config_dir = os.path.join(exp_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)

    return c_logger_dir, tb_logger_dir, evals_dir, output_dir, config_dir


def test_model(test_config, path_config):
    """Test the model using the provided configuration and paths.
    Args:
        - test_config (dict): Configuration parameters for testing, including:
            - 'labels_dmg': List of damage class labels.
            - 'labels_bld': List of building class labels.
            - 'weights_seg': List of weights for segmentation classes.
            - 'weights_damage': List of weights for damage classes.
            - 'weights_loss': List of weights for different loss components.
            - 'mode': Mode of the model ('dmg' or 'bld').
            - 'init_learning_rate': Initial learning rate.
            - 'device': Device to use ('cpu' or 'cuda').
            - 'epochs': Number of epochs.
            - 'batch_size': Batch size for data loading.
            - 'num_chips_to_viz': Number of chips to visualize.
        - path_config (dict): Paths required for testing, including:
            - 'exp_name': Name of the experiment.
            - 'out_dir': Output directory for results.
            - 'shard_splits_json': Path to the JSON file with shard splits.
            - 'label_map_json': Path to the JSON file with label mappings.
            - 'starting_checkpoint_path': Path to the checkpoint to resume from.
    """
    # setup output directories
    c_log_dir, tb_logger_dir, evals_dir, output_dir, config_dir = output_directories(
        path_config['out_dir'], path_config['exp_name'])
    dump_json(os.path.join(config_dir, 'test_config.txt'), test_config)
    dump_json(os.path.join(config_dir, 'test_path_config.txt'), path_config)

    logger_test = SummaryWriter(log_dir=tb_logger_dir)
    log = LoggerSingleton("Testing Model", c_log_dir)

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

    # TEST CONFIG

    # define model
    model = SiamUnet().to(device=device)
    log.info(model.model_summary())

    # resume from a checkpoint if provided
    epoch = resume_model(model, path_config['starting_checkpoint_path'],
                         tb_logger_dir, test_config)

    # define loss functions and weights on classes
    weights_seg_tf = torch.FloatTensor(test_config['weights_seg'])
    weights_damage_tf = torch.FloatTensor(test_config['weights_damage'])
    weights_loss = test_config['weights_loss']

    # loss functions
    criterion_seg_1 = nn.CrossEntropyLoss(
        weight=weights_seg_tf).to(device=device)
    criterion_seg_2 = nn.CrossEntropyLoss(
        weight=weights_seg_tf).to(device=device)
    criterion_damage = nn.CrossEntropyLoss(
        weight=weights_damage_tf).to(device=device)

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

    # TEST STEP
    test = Phase(test_context, static_context)

    # last epoch
    epoch_context = {
        'epoch': epoch,
        'epochs': epoch,
        'step_tr': None,
        'model': model,
        'optimizer': None,
        'save_path': output_dir
    }

    with torch.no_grad():
        test_metrics, test_loss = test.run_epoch(epoch_context)

    log.info(f"test loss:{test_loss};")

    # save evalution metrics
    for key, met in test_metrics.items():
        met.to_csv(os.path.join(evals_dir, f'{key}.csv'), index=False)

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
        'shard_splits_json': \
                    '/home/mrtc101/Desktop/tesina/repo/my_siames/data/xBD/splits/shard_splits.json',
        'label_map_json': \
                    '/home/mrtc101/Desktop/tesina/repo/my_siames/data/constants/xBD_label_map.json',
        'starting_checkpoint_path': None
    }
    test_model(test_config, path_config)
