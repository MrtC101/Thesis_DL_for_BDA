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
        log = LoggerSingleton()
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
                self.logger.add_scalars(f'{self.phase}/learning_rate',
                                        {"lr": optimizer.param_groups[0]["lr"]},
                                         epoch)
            log = LoggerSingleton(f"{self.phase} Step")
            log.info(f'{self.phase.upper()}  epoch: {epoch}/{epochs}')
            start_time = datetime.now()

            result = func(self, **args)
            
            duration = datetime.now() - start_time
            self.logger.add_scalar(f'{self.phase}/time', duration.total_seconds(), epoch)

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
        log = LoggerSingleton()
        confusion_matrices = []

        loss_manager = LossManager(self.weights_loss,self.criterions)
        
        for batch_idx, data in enumerate(tqdm(self.loader)):
            log.info(f"Step: {batch_idx+1}/{len(self.loader)}")
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
            
            step_matrices = self.metric_manager.\
                compute_confusion_matrices(y_seg, y_cls, pred_masks[0], pred_masks[2], batch_idx)
            confusion_matrices.append(step_matrices)
            
            if (self.phase == "test"):
                self.save_pred(pred_masks[2], batch_idx, save_path)

        loss_manager.log_losses(self.logger, self.phase, epoch)

        self.viz.tb_log_images(self.logger, self.phase, self.dataset, self.sample_ids,\
                                model, epoch, self.device)
        
        confusion_matrices_df = pd.DataFrame(confusion_matrices)
        metrics = self.metric_manager.\
            compute_epoch_metrics(self.phase, self.logger, epoch, confusion_matrices_df)
        return metrics, loss_manager.combined_losses.avg