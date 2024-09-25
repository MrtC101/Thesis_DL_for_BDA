import torch

from models.trainable_model import TrainModel
from training.model_train.epoch_manager import EpochManager
from training.train_pipeline import set_threads
from utils.common.pathManager import FilePath
from utils.dataloaders.train_dataloader import TrainDataLoader
from utils.datasets.train_dataset import TrainDataset
from utils.loggers.console_logger import LoggerSingleton
from utils.loggers.tensorboard_logger import TensorBoardLogger
from utils.metrics.curve_computer import pixel_metric_curves
from utils.metrics.loss_manager import LossManager
from utils.metrics.metric_manager import MetricManager


def inference_on_test(configs: dict, paths: dict):
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
    # PATHS
    out_dir = FilePath(paths['out_dir']).create_folder()
    log_out = out_dir.join("console_logs")
    predicted_dir = out_dir.join("test_pred_masks").create_folder()
    weights_path = out_dir.join('checkpoints', "model_best.pth.tar")
    tb_logger_dir = out_dir.join('test_tb_logs').create_folder()
    metric_dir = out_dir.join('best_metrics').create_folder()

    # loggers
    log = LoggerSingleton("MODEL_ON_TEST", folder_path=log_out)

    tb_logger = TensorBoardLogger(tb_logger_dir, configs['num_chips_to_viz'])

    # Load DATA
    xBD_test = TrainDataset('test', paths['split_json'], paths['mean_json'])
    log.info(f'xBD_disaster_dataset test length: {len(xBD_test)}')
    test_loader = TrainDataLoader(xBD_test,
                                  batch_size=configs['batch_size'], shuffle=False,
                                  num_workers=configs['batch_workers'], pin_memory=False
                                  )

    # Device & Model
    set_threads()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrainModel().to(device=device)

    log.info(f'Loading checkpoint from {weights_path}')
    optimizer, starting_epoch, best_acc = model.load_freezed_weights(weights_path, device)

    log.info(f"Loaded checkpoint, starting epoch is {starting_epoch}, best f1 is {best_acc}")
    epochs = configs['tot_epochs']

    # loss functions & managers
    w_seg = torch.FloatTensor(configs['weights_seg'])
    w_damage = torch.FloatTensor(configs['weights_dmg'])
    criterion_seg = torch.nn.CrossEntropyLoss(weight=w_seg).to(device=device)
    criterion_damage = torch.nn.CrossEntropyLoss(weight=w_damage).to(device=device)
    criterions = [criterion_seg, criterion_seg, criterion_damage]
    loss_manager = LossManager(configs['weights_loss'], criterions)
    metric_manager = MetricManager(configs['labels_bld'], configs['labels_dmg'])

    with tb_logger, torch.no_grad():
        testing_manager = EpochManager(
            mode=EpochManager.Mode.TESTING,
            loader=test_loader,
            loss_manager=loss_manager,
            metric_manager=metric_manager,
            tb_logger=tb_logger,
            tot_epochs=epochs,
            optimizer=optimizer,
            model=model,
            device=device
        )

        # TESTING
        test_metrics, test_loss = testing_manager.run_epoch(1, predicted_dir)
        MetricManager.save_metrics([test_metrics], [{"epoch": 1, "loss": test_loss}],
                                   metric_dir, "test")
        log.info(f"Testing Loss: {test_loss:.3f}")
        pixel_metric_curves(test_loader, model, device, metric_dir)
        mean = test_metrics["dmg_pixel_level"]["f1_harmonic_mean"].mean()
        log.info(f"Harmonic mean for f1 for test: {mean}")
