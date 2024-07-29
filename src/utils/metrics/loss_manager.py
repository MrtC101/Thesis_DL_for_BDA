# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
from utils.metrics.common import AverageMeter
import torch


class LossManager():

    def __init__(self, wights, criterions):
        self.weights = wights
        self.criterions = criterions
        self.combined_losses = AverageMeter()
        self.losses_seg_pre = AverageMeter()
        self.losses_seg_post = AverageMeter()
        self.losses_dmg = AverageMeter()

    def compute_loss(self, predicted_masks: list[torch.Tensor], input_data: torch.Tensor,
                     segmentation_targets: torch.Tensor, mask_targets: torch.Tensor) -> torch.nn.CrossEntropyLoss:
        """Computes loss function"""
        # List of true masks corresponding to the predicted masks
        true_masks = [segmentation_targets,
                      segmentation_targets, mask_targets]
        # Calculate individual losses for each pair of predicted and true masks
        individual_losses = [
            criterion(pred_mask, true_mask)
            for criterion, pred_mask, true_mask in zip(self.criterions, predicted_masks, true_masks)
        ]
        # Combine individual losses with their corresponding weights
        combined_loss = sum(weight * loss for weight,
                            loss in zip(self.weights, individual_losses))
        # Update the tracked losses
        self.update_losses(
            combined_loss, individual_losses, input_data.size(0))
        return combined_loss

    def update_losses(self, combined_loss, losses, x_pre_size):
        """Updates each AverageMeter"""
        self.combined_losses.update(combined_loss.item(), x_pre_size)
        self.losses_seg_pre.update(losses[0].item(), x_pre_size)
        self.losses_seg_post.update(losses[1].item(), x_pre_size)
        self.losses_dmg.update(losses[2].item(), x_pre_size)

    def log_losses(self, tb_log, phase, epoch):
        obj = {
            'seg_pre': self.losses_seg_pre.avg,
            'seg_post': self.losses_seg_post.avg,
            'dmg': self.losses_dmg.avg,
            'tot_loss': self.combined_losses.avg
        }
        tb_log.add_scalars(f'{phase}/loss', obj, epoch)
