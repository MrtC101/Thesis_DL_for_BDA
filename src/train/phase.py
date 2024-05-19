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
from utils.visualization.raster_label_visualizer import RasterLabelVisualizer
from models.siames.end_to_end_Siam_UNet import SiamUnet
from utils.metrics.loss_manager import LossManager
import pandas as pd

log = LoggerSingleton()

class Phase:
    """
        Class that implementes an iteration from a phase "train" or "val"
    """

    def __init__(self, phase_context, static_context):
        self.__init_phase_context__(**phase_context)
        self.__init_static_context__(**static_context)
        self.metric_manager = MetricManager(phase_context, static_context)
        self.viz = RasterLabelVisualizer(label_map=static_context['label_map_json'])

    def __init_phase_context__(self, logger, phase, loader, dataset,
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

    def save_pred(self,pred_dmg_mask: np.ndarray,batch_id, path: str) -> None:
        """Saves current prediction image"""
        log.info('save png image for damage level predictions: ' + path)
        os.makedirs(path,exist_ok=True)
        for i in range(pred_dmg_mask.shape[0]):
            file = os.path.join(path,f"batch_{batch_id}_{i}_dmg_mask.png")
            arr = np.array(pred_dmg_mask[i, :, :]).astype(np.uint8)
            cv2.imwrite(file,arr)
        log.info(f'saved image size: {pred_dmg_mask.size()}')

    def logging_wrapper(func):
        """Wrapper applied to the epoch_iteration method
          for printing messages"""
        def decorator(self, args, **kwargs):
            optimizer = args['optimizer']
            epochs = args['epochs']
            epoch = args['epoch']

            if (self.phase == "train"):
                self.logger.add_scalars(f'{self.phase}',{"lr": optimizer.param_groups[0]["lr"]},
                                         epoch)

            log.info(f'epoch: {epoch}/{epochs}')
            start_time = datetime.now()

            result = func(self, **args)
            
            duration = datetime.now() - start_time
            self.logger.add_scalar(f'time_{self.phase}', duration.total_seconds(), epoch)

            return result
        return decorator

    @logging_wrapper
    def run_epoch(self, model : SiamUnet, optimizer, epoch, save_path, **kwargs):
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
        confusion_matrices = []

        loss_manager = LossManager(self.weights_loss,self.criterions)
        
        for batch_idx, data in enumerate(tqdm(self.loader)):
            log.info(f"Step: {batch_idx}")
            #STEP
            # move to device, e.g. GPU
            x_pre = data['pre_image'].to(device=self.device)
            x_post = data['post_image'].to(device=self.device)
            y_seg = data['building_mask'].to(device=self.device)
            y_cls = data['damage_mask'].to(device=self.device)

            if (self.phase == "train"):
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()

            logit_masks = model(x_pre, x_post)

            loss = loss_manager.compute_loss(logit_masks,x_pre,y_seg,y_cls)

            if (self.phase == "train"):
                loss.backward()  # compute gradients
                optimizer.step()

            # Compute predictions
            softmax = torch.nn.Softmax(dim=1)
            pred_masks = [torch.argmax(softmax(logit_mask), dim=1) for logit_mask in logit_masks]
            
            step_matrices = \
                self.compute_confusion_matrices(y_seg, y_cls, pred_masks[0], pred_masks[2], batch_idx)
            confusion_matrices.append(step_matrices)
            
            if (self.phase == "test"):
                self.save_pred(pred_masks[2], batch_idx, save_path)

        loss_manager.log_losses(self.logger, self.phase, epoch)

        self.viz.prepare_for_vis(self.logger, self.phase, self.dataset, self.sample_ids, model,
                                  epoch, self.device)
        
        log.info(f'Compute actual metrics for model evaluation based on {self.phase} set ...')
        confusion_matrices_df = pd.DataFrame(confusion_matrices)
        metrics = self.compute_epoch_metrics(epoch, confusion_matrices_df)
        self.log_metrics(metrics)
        return metrics, loss_manager.combined_losses.avg
 
    def compute_confusion_matrices( self, y_seg: torch.Tensor, y_cls: torch.Tensor, 
                                    pred_y_seg: torch.Tensor, pred_y_cls: torch.Tensor,
                                    batch_idx: int, *kwargs ) -> dict:
        """
        Computes confusion matrices for damage and building classification at different levels.
        
        Args:
            y_seg (torch.Tensor): Ground truth segmentation tensor.
            y_cls (torch.Tensor): Ground truth classification tensor.
            pred_y_seg (torch.Tensor): Predicted segmentation tensor.
            pred_y_cls (torch.Tensor): Predicted classification tensor.
            batch_idx (int): Index of the current batch.
            *kwargs: Additional arguments.

        Returns:
            dict: Dictionary containing confusion matrices and batch identifier.
        """
        func = self.metric_manager.get_confusion_matrices_for
        levels = [Level.PX_DMG, Level.PX_BLD, Level.OBJ_DMG, Level.OBJ_BLD]
        matrices_keys = ["px_dmg_matrices", "px_bld_matrices",
                          "obj_dmg_matrices", "obj_bld_matrices"]
        matrices = {}
        for key, lvl in zip(matrices_keys, levels):
            matrices[key] = func(lvl, y_seg, y_cls, pred_y_seg, pred_y_cls)
            matrices[key].insert(0, "batch_id", batch_idx)
        return matrices

    def compute_epoch_metrics(self, epoch, confusion_matrices_df : pd.DataFrame):
        """Computes metrics for damage and building classification 
        for the current phase in the current epoch.
        
        Args:
            confusion_matrices (dict): Dictionary containing confusion matrices. 
            epoch (int): The current epoch number.

        Returns:
            dict: Dictionary containing computed metrics for damage and building classification.
        """
        func = self.metric_manager.compute_metrics_for
        metrics_keys = ["dmg_pixel_level", "bld_pixel_level",
                         "dmg_object_level", "bld_object_level"]
        levels = [Level.PX_DMG, Level.PX_BLD, Level.OBJ_DMG, Level.OBJ_BLD]
        matrices_keys = ["px_dmg_matrices", "px_bld_matrices",
                          "obj_dmg_matrices", "obj_bld_matrices"]
        metrics = {}
        for key, lvl, mtrx in zip(metrics_keys,levels,matrices_keys):
            metrics[key] = func(lvl,confusion_matrices_df[mtrx])
            metrics[key].insert(0,"epoch",epoch)
        return metrics 

    def log_metrics(self, metrics: dict[pd.DataFrame]):
        """Logs evaluation metrics using the provided logger."""
        metric_df : pd.DataFrame
        log.info(f"--{self.phase.upper()} METRICS--")        
        for key, metric_df in metrics.items():
            log.info(f"-{key.upper()} METRICS-")
            for index, row in metric_df.iterrows():
                msg = f"{self.phase}_{key}_metrics"
                self.logger.add_scalars(msg, dict(row), row["epoch"])
                log.info(f"{row}")