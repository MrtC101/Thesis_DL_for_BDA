# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

from utils.metrics.curve_computer import make_metric_curves


if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))


from models.trainable_model import TrainModel
from train.epoch_manager import EpochManager
from utils.common.files import dump_json
from utils.dataloaders.train_dataloader import TrainDataLoader
from utils.metrics.loss_manager import LossManager
from utils.metrics.metric_manager import MetricManager
from utils.loggers.console_logger import LoggerSingleton
from utils.loggers.tensorboard_logger import TensorBoardLogger

log = LoggerSingleton()

def save_if_best(metrics_df, best_acc, checkpoint_dir, model, optimizer, epoch):
    """Compares f1_harmonic_mean from pixel level damage metrics and 
    f1_harmonic_mean from object level damage classification metrics and 
    saves the checkpoint as best model if needed"""
    # saves the model with the highest f1_score for damage classification
    # compute average accuracy across all classes to select the best model
    pixel_h_f1 = metrics_df["dmg_pixel_level"]["f1_harmonic_mean"].mean()
    is_best = pixel_h_f1 >= best_acc
    best_acc = pixel_h_f1 if is_best else best_acc

    log.info(
        f'Saved checkpoint for epoch {epoch}. Highest f1 checkpoint so far: {is_best}\n')

    model.save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_f1_avg': pixel_h_f1,
        'best_f1': best_acc
    }, is_best, checkpoint_dir)
    return best_acc

def resume_model(model: TrainModel, checkpoint_path, device,
                  init_learning_rate, tb_logger, new_optimizer):
    
    """Calls the corresponding model resume method"""
    if checkpoint_path and os.path.isfile(checkpoint_path):
        log.info('Loading checkpoint from {}'.format(checkpoint_path))
        return model.resume_from_checkpoint(checkpoint_path, device, init_learning_rate,
                                             tb_logger, new_optimizer)
    else:
        log.info('No valid checkpoint is provided. Start to train from scratch...')
        return model.resume_from_scratch(init_learning_rate)

def output_directories(output_folder_path):
    """Create directories for the current experiment"""
    os.makedirs(output_folder_path, exist_ok=True)

    checkpoint_dir = os.path.join(output_folder_path, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    tb_logger_dir = os.path.join(output_folder_path, 'tb_logs')
    os.makedirs(tb_logger_dir, exist_ok=True)

    config_dir = os.path.join(output_folder_path, 'configs')
    os.makedirs(config_dir, exist_ok=True)

    metric_dir = os.path.join(output_folder_path, 'metrics')
    os.makedirs(metric_dir, exist_ok=True)

    return checkpoint_dir, tb_logger_dir, config_dir, metric_dir

def train_model( train_loader: TrainDataLoader, val_loader: TrainDataLoader,
                output_folder_path : str, configs: dict[str],
                test_loader : TrainDataLoader = None) -> float:
    """Trains the model using the specified configurations.

        Args:
            configs (dict): All the configuration parameters:
                'init_learning_rate': Initial learning rate.
                'tot_epochs': Number of epochs.
                'batch_size': Batch size for data loading.
                'weights_seg': List of weights for segmentation classes.
                'weights_damage': List of weights for damage classes.
                'weights_loss': List of weights for different loss components.
                'device': Device to use ('cpu' or 'cuda').
                'torch_thread' : Threads numbers for torch backend
                'torch_op_threads' : Thread number for torch operations.
                'batch_workers' : Workers number for data loaders.
                'labels_dmg': List of damage class labels.
                'labels_bld': List of building class labels.
                'num_chips_to_viz': Number of chips to visualize.
                'disaster_num': Number of disasters to leave on dataset
                'border_width': Border width for new mask creation.
                'exp_folder_path': path for the current out folder,
                'split_json_path': Path to the JSON file with splits of patches.
                'statistics_json_path': Path to the JSON file with mean and standard deviation.
                'starting_checkpoint_path': Path to the checkpoint to resume from.
                'configuration_num': Number of the set of parameters for this experiment.
                'new_optimizer' : Boolean that indicates if the optimizer saved in 
                checkpoint is ignored
    """
    log = LoggerSingleton("Training Model", 
                         folder_path=os.path.join(output_folder_path, "console_logs"))
    
    # setup output directories
    checkpoint_dir, tb_logger_dir, config_dir, metric_dir = output_directories(output_folder_path)
    dump_json(os.path.join(config_dir, 'configs.txt'), configs)

    # Device  
    device = torch.device(configs['device'] if torch.cuda.is_available() else "cpu")
    log.info(f'Using device: {device}.')

    # Model
    model = TrainModel().to(device=device)
    # log.info(model.model_summary())

    # samples are for tensorboard visualization of same images through epochs
    tb_logger = TensorBoardLogger(tb_logger_dir, configs['num_chips_to_viz'])
    
    # resume from a checkpoint if provided
    optimizer, starting_epoch, best_acc = resume_model(model,configs['starting_checkpoint_path'],
                                                        device,
                                                        configs['init_learning_rate'],
                                                        tb_logger,
                                                        configs['new_optimizer'])
    log.info(f'Loaded checkpoint, starting epoch is {starting_epoch}, best f1 is {best_acc}')

    # loss functions
    weights_seg_tf = torch.FloatTensor(configs['weights_seg'])
    weights_damage_tf = torch.FloatTensor(configs['weights_damage'])
    weights_loss = configs['weights_loss']

    criterion_seg_1 = nn.CrossEntropyLoss(weight=weights_seg_tf).to(device=device)
    criterion_seg_2 = nn.CrossEntropyLoss(weight=weights_seg_tf).to(device=device)
    criterion_damage = nn.CrossEntropyLoss(weight=weights_damage_tf).to(device=device)

    loss_manager = LossManager(weights_loss, [criterion_seg_1,criterion_seg_2,criterion_damage])
    
    metric_manager = MetricManager(configs['labels_bld'],configs['labels_dmg'])

    # scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2000)

    epochs = configs['tot_epochs']
    sheared_vars = {
        'loss_manager' : loss_manager,
        'metric_manager' : metric_manager,
        'tb_logger': tb_logger,
        'tot_epochs': epochs,
        'optimizer': optimizer,
        'model': model,
        'device': device,
    }
    
    # Objects for training
    training = EpochManager(mode=EpochManager.Mode.TRAINING, loader=train_loader, **sheared_vars)
    validation = EpochManager(mode=EpochManager.Mode.VALIDATION, loader=val_loader, **sheared_vars)

    # Metrics
    train_metrics = []
    val_metrics = []
    loss_metrics = []

    for epoch in trange(starting_epoch, epochs+1, desc=f"Epoch"):
        # TRAINING
        train_epoch_metrics, tr_loss = training.run_epoch(epoch)
        train_metrics.append(train_epoch_metrics)
        # VALIDATION
        with torch.no_grad():
            val_epoch_metrics, val_loss = validation.run_epoch(epoch)
        val_metrics.append(val_epoch_metrics)
        
        scheduler.step(val_loss)  # decay Learning Rate

        log.info(f"epoch {epoch}/{configs['tot_epochs']}:\
                  train loss:{tr_loss:3f};\
                  val loss:{val_loss:3f};")
        loss_metrics.append({"epoch":epoch,
                             "tr_loss": tr_loss,
                             "val_loss": val_loss})
        # CHECKPOINT
        best_acc = save_if_best(val_epoch_metrics, best_acc,
                                checkpoint_dir, model, optimizer, epoch)

    MetricManager.save_loss(loss_metrics, metric_dir, "train_loss.csv")
    MetricManager.save_metrics(train_metrics, metric_dir, "train")
    MetricManager.save_metrics(val_metrics, metric_dir, "val")
    
    # TESTING
    if(test_loader is not None):
        predicted_dir = os.path.join(output_folder_path,"test_pred_masks")
        os.makedirs(predicted_dir,exist_ok=True)
        testing = EpochManager(mode=EpochManager.Mode.TESTING, loader=test_loader, **sheared_vars)
        with torch.no_grad():
            test_metrics, test_loss = testing.run_epoch(1,predicted_dir)
        log.info(f"Loss over testing split:{test_loss:3f};")
        MetricManager.save_loss([test_loss], metric_dir,"test_loss.csv")
        MetricManager.save_metrics([test_metrics], metric_dir, "test")
        best_acc = test_metrics["dmg_pixel_level"]["f1_harmonic_mean"].mean()
        make_metric_curves(test_loader, model, metric_dir)
        
    tb_logger.flush()
    tb_logger.close()    
    return best_acc

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
        'shard_splits_json': '/home/mrtc101/Desktop/tesina/repo/my_siames/ \
                data/xBD/splits/shard_splits.json',
        'label_map_json': '/home/mrtc101/Desktop/tesina/repo/my_siames/ \
            data/constants/xBD_label_map.json',
        'starting_checkpoint_path': None
    }
    train_model(train_config, path_config)
