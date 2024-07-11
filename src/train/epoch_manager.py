import os
import sys
import enum
import torch
from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.dataloaders.train_dataloader import TrainDataLoader
from utils.loggers.console_logger import LoggerSingleton
from utils.loggers.tensorboard_logger import TensorBoardLogger
from models.trainable_model import SiamUnet
from utils.metrics.metric_manager import Level, MetricManager
from utils.metrics.loss_manager import LossManager
from utils.datasets.predicted_dataset import PredictedDataset

@dataclass(init=True, repr=False, eq=False, order=False, unsafe_hash=False, frozen=False,
           match_args=True, kw_only=False, slots=False, weakref_slot=False)
class EpochManager:
    """Manages a single epoch of training or evaluation.
    Because is instance of dataclass dunder methods are generated automatically"""
    
    class Mode(enum.Enum):
        TRAINING = "train"
        VALIDATION = "val"
        TESTING = "test"
    
    mode : Mode
    loader : TrainDataLoader
    tot_epochs : int
    optimizer : torch.optim.Optimizer
    model : SiamUnet
    device : torch.device
    loss_manager : LossManager
    metric_manager : MetricManager
    tb_logger : TensorBoardLogger
    
    def logging_wrapper(func):
        """Wrapper applied to the epoch_iteration method
          for printing messages"""

        def decorator(*args):
            self = args[0]
            epoch = args[1]
            if (self.mode.value == "train"):
                self.tb_logger.add_scalars(
                    f'{self.mode.value}/learning_rate',
                    {"lr": self.optimizer.param_groups[0]["lr"]},
                    epoch
                )
            log = LoggerSingleton(name=f"{self.mode.value} Step")
            log.info(f'{self.mode.value.upper()}  epoch: {epoch}/{self.tot_epochs}')
            start_time = datetime.now()

            result = func(*args)

            duration = datetime.now() - start_time
            self.tb_logger.add_scalar(
                f'{self.mode.value}/time',
                duration.total_seconds(),
                epoch
            )
            return result
        return decorator

    @logging_wrapper
    def run_epoch(self, epoch, save_path : str = None):
        """Implements the loop for the current epoch and iterates over steps.

        The behavior of this method depends on the `self.mode` variable.
        Additionally, it computes the corresponding confusion matrix for
        each step in one epoch.

        Args:
            epoch (int): The current epoch.
            save_path (str) : Path to a folder to save the patches predicted.
        
        Returns:
            tuple: A tuple containing the confusion matrices, average losses,
             and damage building confusion matrix.
        """
        log = LoggerSingleton()
        confusion_matrices = []
        binary_masks = []
        for batch_idx, (dis_id,tile_id,patch_id,patch) in enumerate(tqdm(self.loader,desc="Step")):
            # STEP
            log.info(f"Step: {batch_idx+1}/{len(self.loader)}")
            # move to device, e.g. GPU
            x_pre = patch['pre_img'].to(device=self.device)
            x_post = patch['post_img'].to(device=self.device)
            y_seg = patch['bld_mask'].to(device=self.device)
            y_cls = patch['dmg_mask'].to(device=self.device)

            if (self.mode == self.mode.TRAINING):
                self.model.train()
                self.optimizer.zero_grad()
            else:
                self.model.eval()

            logit_masks = self.model(x_pre, x_post)

            loss = self.loss_manager.compute_loss(logit_masks, x_pre, y_seg, y_cls)

            if (self.mode == self.mode.TRAINING):
                loss.backward()  # compute gradients
                self.optimizer.step()

            # Compute predictions
            pred_masks = self.model.compute_predictions(logit_masks) 
 
            step_matrices = self.metric_manager.compute_confusion_matrices(
                                                batch_idx,
                                                gt_bld_mask=y_seg,
                                                gt_dmg_mask=y_cls,
                                                pd_bld_mask=pred_masks[0],
                                                pd_dmg_mask=pred_masks[2],
                                                levels=[Level.PX_DMG, Level.PX_BLD]
                                                )
            confusion_matrices.append(step_matrices)

            if (save_path is not None):
                PredictedDataset.save_pred_patch(pred_masks[2], batch_idx, dis_id,
                                              tile_id, patch_id, save_path)
                log.info(f'Prediction with size {pred_masks[2].size()} saved: {save_path}')

        self.loss_manager.log_losses(self.tb_logger, self.mode.value, epoch)
        self.tb_logger.tb_log_images(self.mode.value, self.loader, self.model, epoch, self.device)
        
        metrics = self.metric_manager.compute_epoch_metrics(epoch, confusion_matrices,
                                                            levels=[Level.PX_DMG, Level.PX_BLD]
                                                            )
        self.metric_manager.log_metrics(phase=self.mode.value,
                                        tb_log=self.tb_logger,
                                        metrics=metrics)
        return metrics, self.loss_manager.combined_losses.avg