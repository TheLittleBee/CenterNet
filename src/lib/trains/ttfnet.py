from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import numpy as np

from models.utils import _sigmoid
from .base_trainer import BaseTrainer
from models.losses import FocalLoss
from .fcos import IOULoss


class TTFLoss(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.alpha = 0.54
        self.beta = 0.54
        self.crit = FocalLoss()
        self.wh_crit = IOULoss('linear_iou')

        self.opt = opt

    def forward(self, output, batch):
        hm, wh = output[-1]
        hm = _sigmoid(hm)
        wh = wh * 4 * self.opt.down_ratio
        heatmap, box_target, reg_weight = self.prepare_target(batch)
        hm_loss = self.crit(hm, heatmap)

        pos_inds = reg_weight > 0
        avg_factor = reg_weight.sum().clamp(min=1)

        wh = wh.permute(0, 2, 3, 1)
        if pos_inds.sum() > 0:
            wh_loss = self.wh_crit(wh[pos_inds], box_target[pos_inds], reg_weight[pos_inds]) / avg_factor
        else:
            wh_loss = wh[pos_inds].sum()
        assert not torch.isnan(hm_loss) and not torch.isnan(wh_loss), 'hm {}; wh {}; heatmap {}; box_target {}; reg_weight {}'.format(
            torch.isnan(hm).sum().item(),torch.isnan(wh).sum().item(),torch.isnan(heatmap).sum().item(),torch.isnan(box_target).sum().item(),torch.isnan(reg_weight).sum().item()
        )
        loss = hm_loss + wh_loss
        loss_stats = {'loss': loss, 'hm': hm_loss, 'wh': wh_loss}
        return loss, loss_stats

    def gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        assert sigma_x != 0 and sigma_y != 0
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = heatmap.new_tensor(gaussian)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                          w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def prepare_target(self, batch):
        down_ratio = self.opt.down_ratio
        output_h, output_w = self.opt.input_h // down_ratio, self.opt.input_w // down_ratio
        heatmap_channel = self.opt.num_classes
        target = batch['target']
        mask = batch['mask']
        N = target.size(0)

        heatmap = target.new_zeros((N, heatmap_channel, output_h, output_w))
        fake_heatmap = target.new_zeros((output_h, output_w))
        box_target = target.new_zeros((N, output_h, output_w, 4))
        reg_weight = target.new_zeros((N, output_h, output_w))

        xs = torch.arange(
            0, output_w * down_ratio, step=down_ratio,
            dtype=torch.float32, device=target.device
        )
        ys = torch.arange(
            0, output_h * down_ratio, step=down_ratio,
            dtype=torch.float32, device=target.device
        )
        ys, xs = torch.meshgrid([ys, xs])

        for i in range(N):
            if mask[i].sum() == 0: continue
            gt_boxes = target[i][mask[i]][:, 1:]
            gt_labels = target[i][mask[i]][:, 0]
            boxes_areas_log = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            boxes_areas_log = boxes_areas_log.log()
            boxes_area_topk_log, boxes_ind = torch.sort(boxes_areas_log, descending=True)

            gt_boxes = gt_boxes[boxes_ind]
            gt_labels = gt_labels[boxes_ind]

            feat_gt_boxes = gt_boxes / down_ratio
            feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0, max=output_w - 1)
            feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0, max=output_h - 1)
            feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                                feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

            # we calc the center and ignore area based on the gt-boxes of the origin scale
            # no peak will fall between pixels
            ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                    (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                                   dim=1) / down_ratio).to(torch.int)

            h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
            w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
            if self.alpha != self.beta:
                h_radiuses_beta = (feat_hs / 2. * self.beta).int()
                w_radiuses_beta = (feat_ws / 2. * self.beta).int()

            # larger boxes have lower priority than small boxes.
            for k in range(boxes_ind.shape[0]):
                cls_id = gt_labels[k].long()

                fake_heatmap = fake_heatmap.zero_()
                self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                            h_radiuses_alpha[k].item(), w_radiuses_alpha[k].item())
                heatmap[i, cls_id] = torch.max(heatmap[i, cls_id], fake_heatmap)

                if self.alpha != self.beta:
                    fake_heatmap = fake_heatmap.zero_()
                    self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                                h_radiuses_beta[k].item(),
                                                w_radiuses_beta[k].item())
                box_target_inds = fake_heatmap > 0
                l = xs - gt_boxes[k][0]
                t = ys - gt_boxes[k][1]
                r = gt_boxes[k][2] - xs
                b = gt_boxes[k][3] - ys
                reg_target = torch.stack([l, t, r, b], dim=2)
                box_target[i, box_target_inds] = reg_target[box_target_inds]

                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = local_heatmap.sum() + 1e-7
                local_heatmap *= 2 - boxes_area_topk_log[k] / (boxes_area_topk_log[0] + 1e-7)
                reg_weight[i, box_target_inds] = local_heatmap / ct_div

        return heatmap, box_target, reg_weight


class TTFTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(TTFTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm', 'wh']
        loss = TTFLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError
