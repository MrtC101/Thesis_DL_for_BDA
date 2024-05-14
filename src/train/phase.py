from datetime import datetime
import pandas as pd
import torch
from utils.metrics.common import AverageMeter
from utils.metrics.train_metrics import MetricComputer
from tqdm import tqdm

from utils.visualization.raster_label_visualizer import RasterLabelVisualizer


class Phase:
    """
        Class that implementes an iteration from a phase "train" or val
    """

    def __init__(self, phase_context, static_context):
        self.__init_phase_context__(**phase_context)
        self.__init_static_context__(**static_context)
        self.dmg_metric = MetricComputer(
            static_context['labels_set_dmg'], phase_context, static_context)
        self.bld_metric = MetricComputer(
            static_context['labels_set_bld'], phase_context, static_context)
        self.viz = RasterLabelVisualizer(
            label_map=static_context['label_map_json'])

    def __init_phase_context__(self, logger, phase, loader, dataset, sample_ids,**kwargs):
        self.logger = logger
        self.phase = phase
        self.loader = loader
        self.dataset = dataset
        self.sample_ids = sample_ids

    def __init_static_context__(self, device, crit_seg_1, crit_seg_2, crit_dmg, weights_loss,**kwargs):
        self.device = device
        self.crit_seg_1 = crit_seg_1
        self.crit_seg_2 = crit_seg_2
        self.crit_dmg = crit_dmg
        self.weights_loss = weights_loss

    def logging_wrapper(func):
        def decorator(self, *args, **kwargs):
            optimizer = args[0]['optimizer']
            epochs = args[0]['epochs']
            epoch = args[0]['epoch']

            if (self.phase == "train"):
                self.logger.add_scalar(
                    'learning_rate', optimizer.param_groups[0]["lr"], epoch)

            self.logger.info(f'Model training for epoch {epoch}/{epochs}')
            start_time = datetime.now()

            result = func(*args)

            duration = datetime.now() - start_time
            self.logger.add_scalar(
                f'time_{self.phase}', duration.total_seconds(), epoch)

            return result
        return decorator

    def update_losses(self, scores, x_pre, y_seg, y_cls, losses, losses_seg_pre, losses_seg_post, losses_dmg):
        # modify damage prediction based on UNet arm
        softmax = torch.nn.Softmax(dim=1)
        preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
        for c in range(0, scores[2].shape[1]):
            scores[2][:, c, :, :] = torch.mul(
                scores[2][:, c, :, :], preds_seg_pre)

        loss_seg_pre = self.crit_seg_1(scores[0], y_seg)
        loss_seg_post = self.crit_seg_2(scores[1], y_seg)
        loss_dmg = self.crit_dmg(scores[2], y_cls)
        loss = self.weights_loss[0] * loss_seg_pre + \
            self.weights_loss[1] * loss_seg_post + \
            self.weights_loss[2] * loss_dmg

        losses.update(loss.item(), x_pre.size(0))
        losses_seg_pre.update(loss_seg_pre.item(), x_pre.size(0))
        losses_seg_post.update(loss_seg_post.item(), x_pre.size(0))
        losses_dmg.update(loss_dmg.item(), x_pre.size(0))
        return loss

    @logging_wrapper
    def epoch_iteration(self, model, optimizer, epoch):

        conf_mtrx_dmg_df = pd.DataFrame()
        conf_mtrx_bld_df = pd.DataFrame()

        losses = AverageMeter()
        losses_seg_pre = AverageMeter()
        losses_seg_post = AverageMeter()
        losses_dmg = AverageMeter()

        losses_dict = {
            'lossses': losses,
            'losses_seg_pre': losses_seg_pre,
            'losses_seg_post': losses_seg_post,
            'losses_dmg': losses_dmg
        }

        for batch_idx, data in enumerate(tqdm(self.loader)):
            # move to device, e.g. GPU
            x_pre = data['pre_image'].to(device=self.device)
            x_post = data['post_image'].to(device=self.device)
            y_seg = data['building_mask'].to(device=self.device)
            y_cls = data['damage_mask'].to(device=self.device)

            if (self.phase == "train"):
                model.train()
                optimizer.zero_grad()
            elif (self.phase == "val"):
                model.eval()

            scores = model(x_pre, x_post)

            loss = self.update_losses(
                scores, x_pre, y_seg, y_cls, **losses_dict)

            if (self.phase == "train"):
                loss.backward()  # compute gradients
                optimizer.step()

            # compute predictions & confusion metrics
            softmax = torch.nn.Softmax(dim=1)
            preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
            preds_seg_post = torch.argmax(softmax(scores[1]), dim=1)
            preds_cls = torch.argmax(softmax(scores[2]), dim=1)

            curr_conf_mtrx_dmg_df = self.dmg_metric.compute_conf_mtrx(
                y_pred_mask=preds_cls, y_dmg_mask=y_cls, y_bld_mask=y_seg,
                epoch=epoch, batch_idx=batch_idx)

            conf_mtrx_dmg_df = pd.concat([conf_mtrx_dmg_df, curr_conf_mtrx_dmg_df], axis=0, ignore_index=True) \
                if len(conf_mtrx_dmg_df) > 0 else curr_conf_mtrx_dmg_df

            curr_conf_mtrx_bld_df = self.bld_metric.compute_conf_mtrx(
                y_pred_mask=preds_seg_pre, y_dmg_mask=None, y_bld_mask=y_seg,
                epoch=epoch, batch_idx=batch_idx)

            conf_mtrx_bld_df = pd.concat([conf_mtrx_bld_df, curr_conf_mtrx_bld_df], axis=0, ignore_index=True) \
                if len(conf_mtrx_bld_df) > 0 else curr_conf_mtrx_bld_df

        self.logger.add_scalars(f'loss_{self.phase}', {
            '_total': losses.avg,
            '_seg_pre': losses_seg_pre.avg,
            '_seg_post': losses_seg_post.avg,
            '_dmg': losses_dmg.avg
        }, epoch)

        self.viz.prepare_for_vis(softmax, self.logger, self.phase, self.dataset, 
                                    self.sample_ids, model, epoch, self.device)

        return conf_mtrx_dmg_df, conf_mtrx_bld_df, losses

    def compute_metrics(self,
                        conf_mtrx_dmg_df: pd.DataFrame,
                        conf_mtrx_bld_df: pd.DataFrame, epoch_context: dict):
        """
            Computes metrics for damage and building classification for current phase in current epoch.
        """
        curr_dmg_metrics, f1_harmonic_mean = \
            self.dmg_metric.compute_metrics_for(epoch_context['epoch'], conf_mtrx_dmg_df)

        curr_bld_metrics, _ = \
            self.bld_metric.compute_metrics_for(epoch_context['epoch'], conf_mtrx_bld_df)

        return curr_dmg_metrics, curr_bld_metrics, f1_harmonic_mean
