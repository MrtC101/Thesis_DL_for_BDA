import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
from utils.metrics.common import AverageMeter
import torch

class LossManager():

    def __init__(self,wights,criterions):
        self.weights = wights
        self.criterions = criterions
        self.combined_losses = AverageMeter()
        self.losses_seg_pre = AverageMeter()
        self.losses_seg_post = AverageMeter()
        self.losses_dmg = AverageMeter()
    
    def compute_loss(self, pred_masks : list[torch.Tensor], x_pre : torch.Tensor,
                    y_seg : torch.Tensor, y_mask : torch.Tensor) -> torch.nn.CrossEntropyLoss:
        """Computes loss function"""
        true_masks = [y_seg, y_seg, y_mask]
        losses = [c(pm,tm) for c, pm, tm in zip(self.criterions, pred_masks, true_masks)]
        combined_loss = sum(weight * loss for weight, loss in zip(self.weights, losses))
        self.update_losses(combined_loss,losses, x_pre.size(0))  
        return combined_loss
    
    def update_losses(self, combined_loss, losses, x_pre_size):
        """Updates each AverageMeter"""
        self.combined_losses.update(combined_loss.item(), x_pre_size)
        self.losses_seg_pre.update(losses[0].item(), x_pre_size)
        self.losses_seg_post.update(losses[1].item(), x_pre_size)
        self.losses_dmg.update(losses[2].item(), x_pre_size)

    def log_losses(self,tb_log,phase,epoch):
        obj = {
            'total': self.combined_losses.avg,
            'seg_pre': self.losses_seg_pre.avg,
            'seg_post': self.losses_seg_post.avg,
            'dmg': self.losses_dmg.avg
        } 
        tb_log.add_scalars(f'{phase}/loss', obj, epoch)