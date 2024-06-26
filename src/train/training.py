import os
import sys

from utils.datasets.train_dataset import TrainDataset
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.files import is_dir, dump_json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.siames.end_to_end_Siam_UNet import SiamUnet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import trange
import torch
from train.phase import Phase
from utils.common.logger import LoggerSingleton

log = LoggerSingleton()


def resume_model(model: SiamUnet, checkpoint_path, tb_log_dir, training_config):
    """Calls the corresponding model resume method"""
    if checkpoint_path and os.path.isfile(checkpoint_path):
        log.info('Loading checkpoint from {}'.format(checkpoint_path))
        optimizer, starting_epoch, best_acc = \
            model.resume_from_checkpoint(checkpoint_path, tb_log_dir,
                                         training_config)
        log.info(
            f'Loaded checkpoint, starting epoch is {starting_epoch}, best f1 is {best_acc}')
    else:
        log.info('No valid checkpoint is provided. Start to train from scratch...')
        optimizer, starting_epoch, best_acc = \
            model.resume_from_scratch(training_config)
    return optimizer, starting_epoch, best_acc


def output_directories(out_dir, exp_name):
    """Create directories for the current experiment"""
    # set up directories (TrainPathManager?)
    is_dir(out_dir)
    exp_dir = os.path.join(out_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    tb_logger_dir = os.path.join(exp_dir, 'tb_logs')
    os.makedirs(tb_logger_dir, exist_ok=True)

    config_dir = os.path.join(exp_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)

    metric_dir = os.path.join(exp_dir, 'training_metrics')
    os.makedirs(metric_dir, exist_ok=True)

    return checkpoint_dir, tb_logger_dir, config_dir, metric_dir


def train_model(train_config: dict, path_config: dict) -> None:
    """Trains the model using the specified configurations.

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
            - 'out_dir': Output directory for results.logger_train
            - 'shard_splits_json': Path to the JSON file with shard splits.
            - 'label_map_json': Path to the JSON file with label mappings.
            - 'starting_checkpoint_path': Path to the checkpoint to resume from.
    Returns:
        None

    Example:
        >>> train_model(train_config, path_config)
    """

    log.name = "Training Model"

    # setup output directories
    checkpoint_dir, tb_logger_dir, config_dir, metric_dir = \
        output_directories(path_config['out_dir'], path_config['exp_name'])
    dump_json(os.path.join(config_dir, 'train_config.txt'), train_config)
    dump_json(os.path.join(config_dir, 'path_config.txt'), path_config)

    # torch device
    log.info(f'Using PyTorch version {torch.__version__}.')
    device = torch.device(
        train_config['device'] if torch.cuda.is_available() else "cpu")
    log.info(f'Using device: {device}.')
    # Establecer el número de threads que TorchScript utilizará
    torch.set_num_threads(train_config['torch_op_threads'])
    # Establecer el número de threads que las librerías internas de PyTorch utilizarán
    torch.set_num_interop_threads(train_config['torch_op_threads'])
    log.info(
        f"Número de threads que TorchScript utilizará: {torch.get_num_threads()}")
    log.info(f"Número de threads que las librerías internas de PyTorch utilizarán:\
              {torch.get_num_interop_threads()}")

    # DATA
    # Load datasets
    xBD_train = TrainDataset('train', path_config['dataset_path'], path_config['statistics_path'])
    log.info('xBD_disaster_dataset train length: {}'.format(len(xBD_train)))

    xBD_val = TrainDataset('val', path_config['dataset_path'], path_config['statistics_path'])
    log.info('xBD_disaster_dataset val length: {}'.format(len(xBD_val)))

    train_loader = DataLoader(xBD_train, batch_size=train_config['batch_size'], shuffle=True,
                              num_workers=train_config['batch_workers'], pin_memory=False)
    val_loader = DataLoader(xBD_val, batch_size=train_config['batch_size'], shuffle=False,
                            num_workers=train_config['batch_workers'], pin_memory=False)

    # samples are for tensorboard visualization of same images through epochs
    logger_train = SummaryWriter(log_dir=tb_logger_dir)
    sample_train_ids = xBD_train.get_sample_images(
        train_config['num_chips_to_viz'])
    logger_val = SummaryWriter(log_dir=tb_logger_dir)
    sample_val_ids = xBD_val.get_sample_images(
        train_config['num_chips_to_viz'])

    # TRAINING CONFIG

    # define model
    model = SiamUnet().to(device=device)
    # log.info(model.model_summary())

    # resume from a checkpoint if provided
    optimizer, starting_epoch, best_acc = \
        resume_model(model, path_config['starting_checkpoint_path'], tb_logger_dir, train_config)

    # loss functions
    weights_seg_tf = torch.FloatTensor(train_config['weights_seg'])
    weights_damage_tf = torch.FloatTensor(train_config['weights_damage'])
    weights_loss = train_config['weights_loss']

    criterion_seg_1 = nn.CrossEntropyLoss(
        weight=weights_seg_tf).to(device=device)
    criterion_seg_2 = nn.CrossEntropyLoss(
        weight=weights_seg_tf).to(device=device)
    criterion_damage = nn.CrossEntropyLoss(
        weight=weights_damage_tf).to(device=device)

    # scheduler
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
        'dataset': xBD_val
    }

    # Objects for training
    training = Phase(train_context, static_context)
    validation = Phase(val_context, static_context)

    epoch = starting_epoch
    epochs = train_config['epochs']
    step = 1

    # Metrics
    train_metrics = []
    val_metrics = []

    for ep in trange(epoch, epochs+1, desc=f"Epoch"):
        # epochs
        epoch_context = {
            'epoch': ep,
            'epochs': epochs,
            'step': step,
            'model': model,
            'optimizer': optimizer,
            'save_path': None
        }
        # TRAINING
        train_epoch_metrics, tr_loss = training.run_epoch(epoch_context)
        train_metrics.append(train_epoch_metrics)
        # VALIDATION
        with torch.no_grad():
            val_epoch_metrics, val_loss = validation.run_epoch(epoch_context)
        scheduler.step(val_loss)  # decay Learning Rate
        val_metrics.append(val_epoch_metrics)
        log.info(
            f"epoch {ep}/{epochs}: train loss:{tr_loss:3f}; val loss:{val_loss:3f};")
        # CHECKPOINT
        best_acc = save_if_best(
            val_epoch_metrics, best_acc, checkpoint_dir, **epoch_context)

    save_metrics(train_metrics, metric_dir)
    save_metrics(val_metrics, metric_dir)

    logger_train.flush()
    logger_train.close()
    logger_val.flush()
    logger_val.close()
    log.info('Done')


def save_metrics(metrics, metric_dir):
    """Save metrics in csv"""
    # save evalution metrics
    for epoch in range(len(metrics)):
        for key, met in metrics[epoch].items():
            mode = "w" if not epoch > 0 else "a"
            header = not epoch > 0
            met.to_csv(os.path.join(
                metric_dir, f'{key}.csv'), mode=mode, header=header, index=False)


def save_if_best(metrics_df, best_acc, checkpoint_dir, model, epoch, optimizer, **kwargs):
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
