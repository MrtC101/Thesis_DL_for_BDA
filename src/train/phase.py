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

    metric: MetricComputer

    def __init__(self, phase_context, static_context):
        self.metric = MetricComputer(phase_context, static_context)
        self.logger = phase_context['logger']
        self.phase = phase_context['phase']
        self.loader = phase_context['loader']
        self.dataset = phase_context['dataset']
        self.sample_ids = phase_context['sample_ids']
        self.device = static_context['device']
        self.crit_seg_1 = static_context['crit_seg_1']
        self.crit_seg_2 = static_context['crit_seg_1']
        self.crit_dmg = static_context['crit_dmg']
        self.labels_set_dmg = static_context['labels_set_dmg']
        self.labels_set_bld = static_context['labels_set_bld']
        self.weights_loss = static_context['weights_loss']
        self.viz = RasterLabelVisualizer(label_map=static_context['label_map_json'])

    def iteration(self, epoch_context):
        cols = ['epoch', 'batch_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total_pixels']
        conf_mtrx_dmg_df = pd.DataFrame(columns=cols)
        conf_mtrx_bld_df = pd.DataFrame(columns=cols)
        losses = AverageMeter()
        losses_seg_pre = AverageMeter()
        losses_seg_post = AverageMeter()
        losses_dmg = AverageMeter()

        for batch_idx, data in enumerate(tqdm(self.loader)):
            # move to device, e.g. GPU
            x_pre = data['pre_image'].to(device=self.device)
            x_post = data['post_image'].to(device=self.device)
            y_seg = data['building_mask'].to(device=self.device)
            y_cls = data['damage_mask'].to(device=self.device)

            assert self.phase == "train" or self.phase == "val", f"ERROR {self.phase}"
            if (self.phase == "train"):
                epoch_context['model'].train()
                epoch_context['optimizer'].zero_grad()
            elif (self.phase == "val"):
                epoch_context['model'].eval()

            scores = epoch_context['model'](x_pre, x_post)

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

            if (self.phase == "train"):
                loss.backward()  # compute gradients
                epoch_context['optimizer'].step()

            # compute predictions & confusion metrics
            softmax = torch.nn.Softmax(dim=1)
            preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
            preds_seg_post = torch.argmax(softmax(scores[1]), dim=1)
            preds_cls = torch.argmax(softmax(scores[2]), dim=1)

            conf_mtrx_dmg_df = self.metric.compute_conf_mtrx(y_pred_mask=preds_cls,
                                                             y_dmg_mask=y_cls,
                                                             y_bld_mask=y_seg,
                                                             labels_set=self.labels_set_dmg,
                                                             conf_mtrx_df=conf_mtrx_dmg_df,
                                                             epoch=epoch_context['epoch'],
                                                             batch_idx=batch_idx)

            conf_mtrx_bld_df = self.metric.compute_conf_mtrx(y_pred_mask=preds_seg_pre,
                                                             y_dmg_mask=None,
                                                             y_bld_mask=y_seg,
                                                             labels_set=self.labels_set_bld,
                                                             conf_mtrx_df=conf_mtrx_bld_df,
                                                             epoch=epoch_context['epoch'],
                                                             batch_idx=batch_idx)

        self.logger.add_scalars(f'loss_{self.phase}', {
            '_total': losses.avg,
            '_seg_pre': losses_seg_pre.avg,
            '_seg_post': losses_seg_post.avg,
            '_dmg': losses_dmg.avg
        }, epoch_context['epoch'])

        #self.viz.prepare_for_vis(softmax, self.logger, self.phase, self.dataset, self.sample_ids, 
        #                        epoch_context['model'], epoch_context['epoch'], self.device)

        return conf_mtrx_dmg_df, conf_mtrx_bld_df, losses

    def compute_metrics(self,
                        dmg_metrics: pd.DataFrame, bld_metrics: pd.DataFrame,
                        conf_mtrx_dmg_df: pd.DataFrame, conf_mtrx_bld_df: pd.DataFrame,
                        epoch_context: dict):
        """
            Computes metrics for damage and building classification for current phase in current epoch.
        """
        curr_dmg_metrics, f1_harmonic_mean = \
            self.metric.compute_metrics_for(
                "dmg", epoch_context, self.labels_set_dmg, conf_mtrx_dmg_df)
        dmg_metrics = pd.concat([dmg_metrics, curr_dmg_metrics], axis=0)

        curr_bld_metrics = \
            self.metric.compute_metrics_for(
                "bld", epoch_context, self.labels_set_bld, conf_mtrx_bld_df)
        bld_metrics = pd.concat([bld_metrics, curr_bld_metrics], axis=0)

        return dmg_metrics, bld_metrics, f1_harmonic_mean
