import os
import sys

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
import cv2
import numpy as np
from tqdm import tqdm
import torch
from datetime import datetime
from utils.common.logger import LoggerSingleton
from utils.metrics.metric_manager import Level, MetricManager
from utils.datasets.train_dataset import TrainDataset
from utils.visualization.raster_label_visualizer import RasterLabelVisualizer
from models.siames.end_to_end_Siam_UNet import SiamUnet
from utils.metrics.loss_manager import LossManager
import pandas as pd


class Phase:
    """
        Class that implementes an iteration from a phase "train" or "val"
    """

    def __init__(self, phase_context, static_context):
        self.__init_phase_context__(**phase_context)
        self.__init_static_context__(**static_context)
        self.metric_manager = MetricManager(phase_context, static_context)
        self.viz = RasterLabelVisualizer(
            label_map=static_context['label_map_json'])

    def __init_phase_context__(self, logger, phase, loader , dataset,
                               sample_ids, **kwargs):
        self.logger = logger
        self.phase = phase
        self.loader = loader
        self.dataset = dataset
        self.sample_ids = sample_ids

    def __init_static_context__(self, device, crit_seg_1, crit_seg_2,
                                crit_dmg, weights_loss, **kwargs):
        self.device = device
        self.criterions = [crit_seg_1, crit_seg_2, crit_dmg]
        self.weights_loss = weights_loss

    def logging_wrapper(func):
        """Wrapper applied to the epoch_iteration method
          for printing messages"""

        def decorator(self, args, **kwargs):
            optimizer = args['optimizer']
            epochs = args['epochs']
            epoch = args['epoch']

            if (self.phase == "train"):
                self.logger.add_scalars(f'{self.phase}/learning_rate',
                                        {"lr": optimizer.param_groups[0]["lr"]},
                                        epoch)
            log = LoggerSingleton(f"{self.phase} Step")
            log.info(f'{self.phase.upper()}  epoch: {epoch}/{epochs}')
            start_time = datetime.now()

            result = func(self, **args)

            duration = datetime.now() - start_time
            self.logger.add_scalar(
                f'{self.phase}/time', duration.total_seconds(), epoch)

            return result
        return decorator

    @logging_wrapper
    def run_epoch(self, model: SiamUnet, optimizer, epoch, save_path, **kwargs):
        """Implements the loop for the current epoch and iterates over steps.

        The behavior of this method depends on the `self.phase` variable,
        which can take the values 'train', 'val', or 'test'.
        Additionally, it computes the corresponding confusion matrix for
        each step in one epoch.

        Args:
            model (torch.nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            epoch (int): The current epoch.
            **kwargs: Additional keyword arguments that are not in use.

        Returns:
            tuple: A tuple containing the confusion matrices, average losses,
             and damage building confusion matrix.
        """
        log = LoggerSingleton()
        confusion_matrices = []

        loss_manager = LossManager(self.weights_loss, self.criterions)

        for batch_idx, (dis_id,tile_id,patch_id,patch) in enumerate(tqdm(self.loader,desc="Step")):
            # STEP
            log.info(f"Step: {batch_idx+1}/{len(self.loader)}")
            # move to device, e.g. GPU
            x_pre = patch['pre_img'].to(device=self.device)
            x_post = patch['post_img'].to(device=self.device)
            y_seg = patch['bld_mask'].to(device=self.device)
            y_cls = patch['dmg_mask'].to(device=self.device)

            if (self.phase == "train"):
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()

            logit_masks = model(x_pre, x_post)

            loss = loss_manager.compute_loss(logit_masks, x_pre, y_seg, y_cls)

            if (self.phase == "train"):
                loss.backward()  # compute gradients
                optimizer.step()

            # Compute predictions
            softmax = torch.nn.Softmax(dim=1)
            pred_masks = [torch.argmax(softmax(logit_mask), dim=1)
                        for logit_mask in logit_masks]

            step_matrices = self.metric_manager.compute_confusion_matrices(
                    y_seg, y_cls, pred_masks[0], pred_masks[2], batch_idx,
                    levels=[Level.PX_DMG, Level.PX_BLD])
            confusion_matrices.append(step_matrices)

            if (self.phase == "test"):
                log = LoggerSingleton()
                log.info('save png image for damage level predictions: ' + save_path)
                self.dataset.\
                    save_pred_patch(pred_masks[2], batch_idx, dis_id,tile_id, patch_id, save_path)
                log.info(f'saved image size: {pred_masks[2].size()}')

        loss_manager.log_losses(self.logger, self.phase, epoch)

        self.viz.tb_log_images(self.logger, self.phase, self.dataset, self.sample_ids,
                               model, epoch, self.device)

        confusion_matrices_df = pd.DataFrame(confusion_matrices)
        metrics = self.metric_manager.compute_epoch_metrics(self.phase, self.logger,
                                                            epoch, confusion_matrices_df,
                                                            levels=[Level.PX_DMG, Level.PX_BLD])
        return metrics, loss_manager.combined_losses.avg
