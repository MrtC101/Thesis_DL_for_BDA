# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import torch


class LossManager():
    """Methods to compute loss"""

    def __init__(self, wights, criterions):
        self.weights = wights
        self.criterions = criterions
        self.combined_losses = AverageMeter()
        self.losses_seg_pre = AverageMeter()
        self.losses_seg_post = AverageMeter()
        self.losses_dmg = AverageMeter()

    def compute_loss(self, predicted_masks: list[torch.Tensor], input_data: torch.Tensor,
                     segmentation_targets: torch.Tensor,
                     mask_targets: torch.Tensor) -> torch.nn.CrossEntropyLoss:
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
        self.update_losses(combined_loss, individual_losses, input_data.size(0))
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


class AverageMeter(object):
    """Computes and stores the average and current value
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """

        Args:
            val: mini-batch loss or accuracy value
            n: mini-batch size
        """
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
